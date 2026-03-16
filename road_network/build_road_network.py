"""
Build a road network from:
- Road shapefile (filtered by road class)
- Traffic lights (as signalized nodes)

Outputs:
- GMNS format files for AequilibraE import
- AequilibraE project with built graphs
- Optional ArcGIS export (GeoPackage)
"""
import os
import shutil
import sys

import geopandas as gpd
import pandas as pd
from aequilibrae import Project

# Handle both module import and direct execution
if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from road_network.config import (
        ROADS_SHAPEFILE, TRAFFIC_LIGHTS_SHAPEFILE, GMNS_DIR,
        AEQUILIBRAE_PROJECT_DIR, ARCGIS_EXPORT_DIR, ALLOWED_ROAD_CLASSES, PROJECT_CRS,
        COST_PER_KM, VALUE_OF_TIME, NODE_CLUSTER_TOLERANCE_M,
    )
    from road_network.nodes import (
        create_nodes_from_roads, load_traffic_lights,
        add_traffic_lights_to_nodes, filter_to_main_component,
        cluster_nearby_nodes, merge_clustered_nodes,
    )
    from road_network.links import create_road_edges
    from road_network.shortest_path import (
        compute_shortest_path, compute_shortest_path_networkx,
        add_generalized_cost_to_edges, compute_path_details,
        export_shortest_path_to_arcgis,
    )
else:
    from .config import (
        ROADS_SHAPEFILE, TRAFFIC_LIGHTS_SHAPEFILE, GMNS_DIR,
        AEQUILIBRAE_PROJECT_DIR, ARCGIS_EXPORT_DIR, ALLOWED_ROAD_CLASSES, PROJECT_CRS,
        COST_PER_KM, VALUE_OF_TIME, NODE_CLUSTER_TOLERANCE_M,
    )
    from .nodes import (
        create_nodes_from_roads, load_traffic_lights,
        add_traffic_lights_to_nodes, filter_to_main_component,
        cluster_nearby_nodes, merge_clustered_nodes,
    )
    from .links import create_road_edges
    from .shortest_path import (
        add_generalized_cost_to_edges, export_shortest_path_to_arcgis,
    )


def load_and_clean_roads(roads_path=None):
    """Load roads shapefile and filter to allowed road classes."""
    path = roads_path or ROADS_SHAPEFILE
    roads = gpd.read_file(path)
    
    print(f"Road classes: {roads['CartoClass'].unique().tolist()}")
    print(f"Original roads: {len(roads)}")
    
    roads_filtered = roads[roads['CartoClass'].isin(ALLOWED_ROAD_CLASSES)].copy()
    roads_filtered.reset_index(drop=True, inplace=True)
    
    if roads_filtered.crs.is_geographic:
        roads_filtered = roads_filtered.to_crs(PROJECT_CRS)
    elif roads_filtered.crs != PROJECT_CRS:
        roads_filtered = roads_filtered.to_crs(PROJECT_CRS)
    
    roads_filtered['link_id'] = range(1, len(roads_filtered) + 1)
    
    print(f"Filtered roads: {len(roads_filtered)}")
    return roads_filtered


def fill_aequilibrae_ba_columns(project):
    """Fill empty _ab/_ba columns with 0 so build_graphs() works."""
    direction_columns = (
        "lanes_ab", "lanes_ba", "speed_ab", "speed_ba",
        "capacity_ab", "capacity_ba", "travel_time_ab", "travel_time_ba",
    )
    links_data = project.network.links.data
    to_fill = [c for c in direction_columns if c in links_data.columns]
    if not to_fill:
        return
    
    conn = getattr(project, "conn", None) or getattr(project, "db_connection", None)
    if conn is None:
        return
    
    try:
        if hasattr(conn, "__enter__"):
            with conn as c:
                for col in to_fill:
                    c.execute(f"UPDATE links SET {col} = 0 WHERE {col} IS NULL OR {col} = ''")
                if hasattr(c, "commit"):
                    c.commit()
        else:
            for col in to_fill:
                conn.execute(f"UPDATE links SET {col} = 0 WHERE {col} IS NULL OR {col} = ''")
            if hasattr(conn, "commit"):
                conn.commit()
    except Exception:
        pass


def save_gmns(nodes_gdf, edges_gdf, gmns_dir=None):
    """Save network to GMNS format for AequilibraE import."""
    out_dir = gmns_dir or GMNS_DIR
    os.makedirs(out_dir, exist_ok=True)
    
    node_file = os.path.join(out_dir, "node.csv")
    link_file = os.path.join(out_dir, "link.csv")
    geometry_file = os.path.join(out_dir, "geometry.csv")
    
    for f in [node_file, link_file, geometry_file]:
        if os.path.exists(f):
            os.remove(f)
    
    # Nodes: include signalized field
    nodes_export = nodes_gdf.copy()
    nodes_export['node_id'] = nodes_export['node_id'].astype(int)
    nodes_export['signalized'] = nodes_export['signalized'].astype(int)
    export_cols = ['node_id', 'x_coord', 'y_coord', 'signalized']
    if 'merged_count' in nodes_export.columns:
        export_cols.append('merged_count')
    nodes_export[export_cols].to_csv(node_file, index=False)
    
    # Links: include speed, travel_time, and generalized_cost
    links_export = edges_gdf.copy()
    links_export['link_id'] = links_export['link_id'].astype(int)
    links_export['from_node_id'] = links_export['from_node_id'].astype(int)
    links_export['to_node_id'] = links_export['to_node_id'].astype(int)
    links_export['directed'] = links_export['directed'].astype(int)
    links_export['geometry_id'] = links_export['link_id']
    links_export['lanes'] = links_export['lanes'].fillna(1).astype(int)
    
    # Include all cost-related columns
    link_columns = ['link_id', 'from_node_id', 'to_node_id', 'directed', 'length',
                    'geometry_id', 'allowed_uses', 'lanes', 'speed_kmh', 
                    'travel_time_min', 'generalized_cost']
    link_columns = [c for c in link_columns if c in links_export.columns]
    links_export[link_columns].to_csv(link_file, index=False)
    
    # Geometry
    geometry_df = edges_gdf[['link_id']].copy()
    geometry_df['geometry_id'] = geometry_df['link_id']
    geometry_df['geometry'] = edges_gdf.geometry.apply(lambda g: g.wkt)
    geometry_df[['geometry_id', 'geometry']].to_csv(geometry_file, index=False)
    
    print(f"GMNS saved: {len(nodes_export)} nodes, {len(links_export)} links")
    return node_file, link_file, geometry_file


def create_aequilibrae_project(node_file, link_file, geometry_file, project_dir=None, srid=26917):
    """Create AequilibraE project and import GMNS network."""
    proj_path = project_dir or AEQUILIBRAE_PROJECT_DIR
    
    if os.path.exists(proj_path):
        shutil.rmtree(proj_path)
    
    project = Project()
    project.new(proj_path)
    print("AequilibraE project created")
    
    project.network.create_from_gmns(
        link_file_path=link_file,
        node_file_path=node_file,
        geometry_path=geometry_file,
        srid=srid
    )
    
    fill_aequilibrae_ba_columns(project)
    project.network.build_graphs()
    
    print(f"Network built: {project.network.count_nodes()} nodes, {project.network.count_links()} links")
    return project


def export_to_arcgis(nodes_gdf, edges_gdf, out_dir=None):
    """Export network to GeoPackage for ArcGIS/QGIS visualization."""
    out_path = out_dir or ARCGIS_EXPORT_DIR
    os.makedirs(out_path, exist_ok=True)
    gpkg_path = os.path.join(out_path, "road_network.gpkg")
    
    # Remove existing file to ensure clean export
    if os.path.exists(gpkg_path):
        os.remove(gpkg_path)
    
    # Reset index to ensure node_id/link_id are preserved as columns
    nodes_export = nodes_gdf.copy().reset_index(drop=True)
    edges_export = edges_gdf.copy().reset_index(drop=True)
    
    # Convert to WGS84 for ArcGIS
    nodes_wgs = nodes_export.to_crs("EPSG:4326")
    edges_wgs = edges_export.to_crs("EPSG:4326")
    
    # Ensure ID columns are integers
    nodes_wgs['node_id'] = nodes_wgs['node_id'].astype(int)
    edges_wgs['link_id'] = edges_wgs['link_id'].astype(int)
    edges_wgs['from_node_id'] = edges_wgs['from_node_id'].astype(int)
    edges_wgs['to_node_id'] = edges_wgs['to_node_id'].astype(int)
    
    nodes_wgs.to_file(gpkg_path, layer="nodes", driver="GPKG")
    edges_wgs.to_file(gpkg_path, layer="links", driver="GPKG", mode="a")
    
    print(f"ArcGIS export: {gpkg_path}")
    print(f"  Nodes: {len(nodes_wgs)}, node_id range: {nodes_wgs['node_id'].min()} - {nodes_wgs['node_id'].max()}")
    print(f"  Links: {len(edges_wgs)}, link_id range: {edges_wgs['link_id'].min()} - {edges_wgs['link_id'].max()}")
    return gpkg_path


def build_network(
    roads_path=None,
    traffic_lights_path=None,
    cluster_tolerance_m=None,
    export_arcgis=True,
    verbose=True
):
    """
    Main workflow: build the road network with signalized intersections.
    
    Parameters:
        roads_path: path to roads shapefile (default: from config)
        traffic_lights_path: path to traffic lights shapefile (default: from config)
        cluster_tolerance_m: distance to merge nearby nodes (default: NODE_CLUSTER_TOLERANCE_M)
                            Set to 0 to disable clustering
        export_arcgis: export to GeoPackage for visualization
        verbose: print progress
    
    Returns:
        project: AequilibraE project with built graphs
        nodes_gdf: GeoDataFrame of nodes (with 'signalized' field)
        edges_gdf: GeoDataFrame of edges (with 'generalized_cost' field)
    """
    if cluster_tolerance_m is None:
        cluster_tolerance_m = NODE_CLUSTER_TOLERANCE_M
    
    if verbose:
        print("\n=== Step 1: Load and clean roads ===")
    roads_gdf = load_and_clean_roads(roads_path)
    
    if verbose:
        print("\n=== Step 2: Create nodes from road intersections ===")
    nodes_data, nodes_dict, node_id = create_nodes_from_roads(roads_gdf)
    nodes_gdf = gpd.GeoDataFrame(nodes_data, geometry='geometry', crs=roads_gdf.crs)
    if verbose:
        print(f"Road intersection nodes: {len(nodes_gdf)}")
    
    if verbose:
        print("\n=== Step 3: Load traffic lights and add to nodes ===")
    lights_path = traffic_lights_path or TRAFFIC_LIGHTS_SHAPEFILE
    lights_gdf = load_traffic_lights(lights_path, roads_gdf.crs)
    if verbose:
        print(f"Traffic lights loaded: {len(lights_gdf)}")
    
    nodes_gdf, nodes_dict, node_id = add_traffic_lights_to_nodes(
        lights_gdf, nodes_gdf, nodes_dict, node_id
    )
    
    signalized_count = (nodes_gdf['signalized'] == 1).sum()
    if verbose:
        print(f"Total nodes before clustering: {len(nodes_gdf)} ({signalized_count} signalized)")
    
    # Cluster nearby nodes (handles divided road intersections)
    if cluster_tolerance_m > 0:
        if verbose:
            print(f"\n=== Step 4: Cluster nearby nodes (tolerance: {cluster_tolerance_m}m) ===")
        nodes_gdf, cluster_ids = cluster_nearby_nodes(nodes_gdf, cluster_tolerance_m)
        nodes_gdf, nodes_dict, coord_remap = merge_clustered_nodes(nodes_gdf, nodes_dict)
        
        signalized_count = (nodes_gdf['signalized'] == 1).sum()
        merged_count = (nodes_gdf['merged_count'] > 1).sum() if 'merged_count' in nodes_gdf.columns else 0
        if verbose:
            print(f"Nodes after clustering: {len(nodes_gdf)} ({signalized_count} signalized, {merged_count} merged)")
    else:
        coord_remap = {}
        if verbose:
            print("\n=== Step 4: Skipping node clustering (disabled) ===")
    
    if verbose:
        print("\n=== Step 5: Create road edges ===")
    next_link_id = 1
    edges_gdf, next_link_id = create_road_edges(roads_gdf, nodes_dict, next_link_id, coord_remap)
    if verbose:
        print(f"Road edges: {len(edges_gdf)}")
    
    if verbose:
        print("\n=== Step 6: Filter to main component ===")
    nodes_gdf, edges_gdf = filter_to_main_component(nodes_gdf, edges_gdf, verbose=verbose)
    
    if verbose:
        print("\n=== Step 7: Compute generalized cost ===")
    edges_gdf = add_generalized_cost_to_edges(edges_gdf, nodes_gdf)
    if verbose:
        print(f"  Cost parameters: ${COST_PER_KM}/km + ${VALUE_OF_TIME}/hour")
        print(f"  Generalized cost range: ${edges_gdf['generalized_cost'].min():.4f} - ${edges_gdf['generalized_cost'].max():.4f}")
    
    # Recount signalized after filtering
    signalized_count = (nodes_gdf['signalized'] == 1).sum()
    if verbose:
        print(f"\nFinal: {len(nodes_gdf)} nodes ({signalized_count} signalized), {len(edges_gdf)} edges")
    
    if verbose:
        print("\n=== Step 8: Save GMNS ===")
    node_file, link_file, geometry_file = save_gmns(nodes_gdf, edges_gdf)
    
    if verbose:
        print("\n=== Step 9: Create AequilibraE project ===")
    project = create_aequilibrae_project(node_file, link_file, geometry_file)
    
    if export_arcgis:
        if verbose:
            print("\n=== Step 10: Export to ArcGIS ===")
        export_to_arcgis(nodes_gdf, edges_gdf)
    
    if verbose:
        print("\n=== Network build complete! ===")
    
    return project, nodes_gdf, edges_gdf


if __name__ == "__main__":
    project, nodes_gdf, edges_gdf = build_network()
    
    print("\n=== Testing shortest path (NetworkX with generalized cost) ===")
    all_node_ids = list(nodes_gdf['node_id'].values)
    orig_node = int(1912)#(all_node_ids[0])
    dest_node = int(2651)#(all_node_ids[min(100, len(all_node_ids) - 1)])
    # orig_node = int(all_node_ids[0])
    # dest_node = int(all_node_ids[min(100, len(all_node_ids) - 1)])
    
    # Shortest path by generalized cost using NetworkX
    result = compute_shortest_path_networkx(
        edges_gdf, orig_node, dest_node, 
        weight_field='generalized_cost'
    )
    
    if result["found"]:
        details = compute_path_details(edges_gdf, result["path_links"])
        print(f"\n  Path Details:")
        print(f"    Distance: {details['total_distance_km']:.2f} km")
        print(f"    Travel time: {details['total_travel_time_min']:.1f} min")
        print(f"    Distance cost: ${details['total_distance_cost']:.2f}")
        print(f"    Time cost: ${details['total_time_cost']:.2f}")
        print(f"    Total generalized cost: ${details['total_generalized_cost']:.2f}")
        
        # Export shortest path to ArcGIS
        print("\n=== Export shortest path to ArcGIS ===")
        export_shortest_path_to_arcgis(
            nodes_gdf, edges_gdf, 
            result["path_nodes"], result["path_links"]
        )
    
    project.close()
