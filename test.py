import geopandas as gpd  # use to read shapefiles, filter by road type, compute lengths
import pandas as pd
import gtfs_kit as gk  # For reading GTFS feeds
# import aequilibrae as ae  # For traffic modeling, shortest path, assignment
from aequilibrae import Project
from shapely.geometry import LineString, Point
import os # operating system to check if file/folder exists
import shutil # shell utilities to delete directory tree
from methods import *
from links_methods import *
from shortest_path import test_shortest_path
import numpy as np

# Question: Do we want to see transit routes AND roads as links between nodes? Yes - because multimodal solution???
# if yes, we need to transofmr roads_filtered into gmns ingestible format and concat in all_edges in step 6
# otherwise we'll just have a network for bus / transit stuff which might be sufficient

"""
Step 1: Import and Clean Roads Data 
"""
print("\n--- Import and clean roads ---")
roads = gpd.read_file("Roads/Roads.shp")
roads_filtered = clean_data(roads)
print("Filtered number of rows:", roads_filtered.shape[0])
# print(roads_filtered.head())


"""
Step 2: Build nodes from roads 
"""
print("\n--- Create nodes from roads ---")
nodes_data, node_id = filter_nodes(roads_filtered)
nodes = {(row['x_coord'], row['y_coord']): row['node_id'] for row in nodes_data}
gmns_nodes = gpd.GeoDataFrame(nodes_data, geometry='geometry', crs=roads_filtered.crs)
print("Initial nodes from roads:", len(gmns_nodes))


"""
Step 3: Import GTFS Data into stops
"""
print("\n--- Import and clean GTFS Data ---")
# Read GTFS
feed = gk.read_feed("GRT_GTFS.zip", dist_units="km")

# Build stops GeoDataFrame
stops_gdf = gpd.GeoDataFrame(
    feed.stops.copy(),
    geometry=gpd.points_from_xy(
        feed.stops.stop_lon,
        feed.stops.stop_lat
    ),
    crs="EPSG:4326"
).to_crs(roads_filtered.crs)

# Cleaning
filtered_stops_gdf = clean_gfts(stops_gdf, roads_filtered)

# Collapse some stops into 1 node if they are in the same 300x300 grid area to simplify number of nodes
filtered_stops_gdf["x"] = filtered_stops_gdf.geometry.x
filtered_stops_gdf["y"] = filtered_stops_gdf.geometry.y

filtered_stops_gdf["cell"] = (
    (filtered_stops_gdf["x"] // 500).astype(int).astype(str)
    + "_"
    + (filtered_stops_gdf["y"] // 500).astype(int).astype(str)
)

# Create mapping: original stop_id -> representative stop_id
stop_to_representative = {}

# Reset index to make stop_id a column BEFORE grouping
filtered_stops_gdf_with_id = filtered_stops_gdf.reset_index()

# Now group and keep the first stop_id in each cell
filtered_stops_gdf_collapsed = filtered_stops_gdf_with_id.groupby("cell").first().reset_index()

# Build the mapping
for cell, group in filtered_stops_gdf_with_id.groupby("cell"):
    representative_id = group.iloc[0]['stop_id']  # Now stop_id is a column
    for stop_id in group['stop_id']:
        stop_to_representative[stop_id] = representative_id

# Use the collapsed version with stop_id as index
filtered_stops_gdf = filtered_stops_gdf_collapsed.set_index('stop_id')

gmns_nodes, nodes, node_id = snap_and_add_stops_to_nodes(
    filtered_stops_gdf, gmns_nodes, nodes, node_id
)

print(f"Clean stops retained: {len(filtered_stops_gdf)}")

"""
Step 3b: Import ION LRT stops and add as nodes (exact coordinates)
"""
print("\n--- Import ION LRT stops ---")
ion_stops_gdf = load_ion_stops("ION_Stops.csv", roads_filtered.crs, stage1="LRT", stop_status="Constructed")
gmns_nodes, nodes, node_id = add_ion_stops_to_nodes(ion_stops_gdf, gmns_nodes, nodes, node_id)
print(f"ION LRT stops added: {len(ion_stops_gdf)}")

# Export to a GeoPackage (better for large datasets)
# export_path = "Export/gmns_nodes_wgs84.gpkg"
# export_to_GIS(export_path)
# gmns_nodes_wgs = gmns_nodes.to_crs(epsg=4326)
# gmns_nodes_wgs.to_file(export_path, layer="nodes", driver="GPKG")

"""
Step 4: Create GMNS Edges/Links
- Create road edges from roads_filtered
- Create transit edges from GTFS stop_times
- Filter out duplicate/overlapping edges
"""
print("\n--- Creating Links ---")
all_edges = []
next_link_id = 1

# Create Road Edges
road_edges_gdf, next_link_id = create_road_edges(roads_filtered, nodes, next_link_id)


# Create Transit Edges
transit_edges_gdf = create_transit_edges(feed, filtered_stops_gdf, nodes, road_edges_gdf, roads_filtered, next_link_id, stop_to_representative)
next_link_id = transit_edges_gdf["link_id"].max() + 1 if len(transit_edges_gdf) > 0 else next_link_id

# Create ION LRT edges (effective length = time-equivalent so LRT is faster than bus)
ion_edges_gdf, next_link_id = create_ion_edges(ion_stops_gdf, nodes, next_link_id, roads_filtered.crs)

# Combine all edges
gmns_edges = pd.concat([road_edges_gdf, transit_edges_gdf, ion_edges_gdf], ignore_index=True)
print(f"Total GMNS Edges: {len(gmns_edges)} (roads: {len(road_edges_gdf)}, transit: {len(transit_edges_gdf)}, ION LRT: {len(ion_edges_gdf)})")

# Keep only the main (largest) weak connected component; drop disconnected nodes and their links
print("Filtering to main component (removing disconnected nodes)...")
gmns_nodes, gmns_edges = filter_network_to_main_component(gmns_nodes, gmns_edges, verbose=True)
print(f"After filter: {len(gmns_nodes)} nodes, {len(gmns_edges)} links")

# Export edges
# export_path_edges = "Export/gmns_edges_wgs84.gpkg"
# export_to_GIS(export_path_edges)
# gmns_edges_wgs = gmns_edges.to_crs(epsg=4326)
# gmns_edges_wgs.to_file(export_path_edges, layer="edges", driver="GPKG")

"""
Step 5: Save GMNS network
"""
print("\n--- Prep GMNS network  ---")
# Prepare GMNS files
gmns_dir = "Network/gmns"
os.makedirs(gmns_dir, exist_ok=True)

node_file = os.path.join(gmns_dir, "node.csv")
link_file = os.path.join(gmns_dir, "link.csv")
geometry_file = os.path.join(gmns_dir, "geometry.csv")

# remove file if exists
for f in [node_file, link_file, geometry_file]:
    if os.path.exists(f):
        os.remove(f)  

# Prepare nodes - AequilibraE expects specific columns
gmns_nodes_export = gmns_nodes.copy()
gmns_nodes_export['node_id'] = gmns_nodes_export['node_id'].astype(int)
gmns_nodes_export[['node_id', 'x_coord', 'y_coord']].to_csv(node_file, index=False)

# Prepare links - ensure all required columns exist
gmns_links_export = gmns_edges.copy()
gmns_links_export['link_id'] = gmns_links_export['link_id'].astype(int)
gmns_links_export['from_node_id'] = gmns_links_export['from_node_id'].astype(int)
gmns_links_export['to_node_id'] = gmns_links_export['to_node_id'].astype(int)
gmns_links_export['directed'] = gmns_links_export['directed'].astype(int)
gmns_links_export['allowed_uses'] = gmns_links_export['allowed_uses']
gmns_links_export['geometry_id'] = gmns_links_export['link_id']
# Lanes required for GMNS import; transit edges may not have it — use 1 where missing
gmns_links_export['lanes'] = gmns_links_export['lanes'].fillna(1).astype(int)

gmns_links_export[['link_id', 'from_node_id', 'to_node_id', 'directed', 'length', 'geometry_id', 'allowed_uses', 'lanes']].to_csv(
    link_file, index=False
)

# Prepare separate geometry.csv file in GMNS format
geometry_df = gmns_edges[['link_id']].copy()
geometry_df['geometry_id'] = geometry_df['link_id']
geometry_df['geometry'] = gmns_edges.geometry.apply(lambda g: g.wkt)  # Convert to WKT
geometry_df = geometry_df[['geometry_id', 'geometry']]  # Only these two columns
geometry_df.to_csv(geometry_file, index=False)


print(f"Exported {len(gmns_nodes_export)} nodes and {len(gmns_links_export)} links to GMNS format")


"""
Step 6: Create project in AequilibraE
"""
print("\n--- Setup aequilibraE project  ---")
project_path = "Network/aequilibrae_project"
if os.path.exists(project_path):
    shutil.rmtree(project_path)

project = Project()
project.new(project_path)
print("✓ Project created successfully")

# Import network from GMNS
print("Importing network from GMNS...")
project.network.create_from_gmns(
    link_file_path=link_file,
    node_file_path=node_file,
    geometry_path=geometry_file,
    srid=26917
)

links_table = project.network.links
print(f"Number of links imported: {len(links_table.data)}")
print(f"\nColumns in AequilibraE links table:")
print(links_table.data.columns.tolist())

# Check if length made it through
if 'length' in links_table.data.columns:
    print(f"\n✓ 'length' column found in AequilibraE!")
    print(f"  Min: {links_table.data['length'].min():.2f}")
    print(f"  Max: {links_table.data['length'].max():.2f}")
    print(f"  Nulls: {links_table.data['length'].isna().sum()}")
else:
    print(f"\n✗ 'length' column NOT found in AequilibraE")
    print(f"Available numeric columns: {links_table.data.select_dtypes(include='number').columns.tolist()}")

fill_aequilibrae_ba_columns(project)

print("Building graphs...")
project.network.build_graphs()

print(f"✓ Network built successfully!")
print(f"  Nodes: {project.network.count_nodes()}")
print(f"  Links: {project.network.count_links()}")


"""
Step 7: Solve shortest path in AequilibraE
"""
print("\n--- Testing Shortest Path ---")

# Choose origin and destination node IDs
all_node_ids = list(gmns_nodes['node_id'].values)
orig_node = int(all_node_ids[0])
dest_node = int(all_node_ids[1400])

# Test shortest path using helper function
result = test_shortest_path(project, orig_node, dest_node)
project.close()

exit()




# Export full network for visualizations
print("\n--- Exporting Network ---")
links_gdf = project.network.links.data
nodes_gdf = project.network.nodes.data

links_gdf_wgs = links_gdf.to_crs(epsg=4326)
nodes_gdf_wgs = nodes_gdf.to_crs(epsg=4326)

links_gdf_wgs.to_file("export_network.gpkg", layer="links", driver="GPKG")
nodes_gdf_wgs.to_file("export_network.gpkg", layer="nodes", driver="GPKG", mode='a')

print("✓ Network exported to export_network.gpkg")

project.close()
print("\n✓ Step 5 complete!")

