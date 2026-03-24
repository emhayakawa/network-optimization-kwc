import geopandas as gpd  # use to read shapefiles, filter by road type, compute lengths
import pandas as pd
from aequilibrae import Project
from shapely.geometry import Point
import os
from collections import defaultdict
from shapely.geometry import Point


def clean_data(roads):
    # Find unique categories in CartoClass
    print(roads['CartoClass'].unique())
    print("Original number of rows:", roads.shape[0])
    print(roads.crs) # coordinate system

    # Filter out Alleyway, Localstreet, Private
    allowed_classes = ['Freeway', 'Highway', 'Ramp', 'Arterial'] #'Collector', 
    roads_filtered = roads[roads['CartoClass'].isin(allowed_classes)].copy()
    roads_filtered.reset_index(drop=True, inplace=True)

    if roads_filtered.crs.is_geographic:
        roads_filtered = roads_filtered.to_crs("EPSG:26917")  

    # check for null vals in distances
    print(roads.columns)
    print(roads['FlowDirect'].unique())
    roads['FlowDirect']

    roads.geometry.isna().sum() #0
    roads.is_empty.sum() #0

    # add a link_id for later use when transforming to GMNS format
    roads_filtered = roads_filtered.copy()
    roads_filtered['link_id'] = range(1, len(roads_filtered)+1)
    return roads_filtered


def snap(coord, precision=3):
    return (round(coord[0], precision), round(coord[1], precision))


def filter_nodes(roads_filtered):
    coord_count = defaultdict(int)

    # count how many road segments touch each coordinate 
    for idx, row in roads_filtered.iterrows():
        coords = list(row.geometry.coords)
        start, end = snap(coords[0]), snap(coords[-1])
        coord_count[start] += 1
        coord_count[end] += 1

    nodes = {}
    node_id = 1

    # only keep node if it is a road intersection 
    for coord, count in coord_count.items():
        if count >= 2:  # intersections
            nodes[coord] = node_id
            node_id += 1

    # Create GeoDataFrame for GMNS nodes
    nodes_data = [{'node_id': nid, 'x_coord': coord[0], 'y_coord': coord[1], 'geometry': Point(coord)}
              for coord, nid in nodes.items()]
    return nodes_data, node_id

def get_main_component_node_ids(edges_df):
    """
    Returns the set of node_ids that belong to the largest weak connected component.
    edges_df must have columns from_node_id and to_node_id.
    Uses undirected connectivity (each link connects both ways for this check).
    """
    a = edges_df["from_node_id"].astype(int)
    b = edges_df["to_node_id"].astype(int)
    all_nodes = sorted(set(a) | set(b))
    node_to_idx = {n: i for i, n in enumerate(all_nodes)}
    n = len(all_nodes)
    adj = [[] for _ in range(n)]
    for i in range(len(edges_df)):
        u, v = node_to_idx[a.iloc[i]], node_to_idx[b.iloc[i]]
        adj[u].append(v)
        adj[v].append(u)
    seen = [False] * n
    components = []
    for start in range(n):
        if seen[start]:
            continue
        comp = []
        stack = [start]
        while stack:
            u = stack.pop()
            if seen[u]:
                continue
            seen[u] = True
            comp.append(all_nodes[u])
            for v in adj[u]:
                if not seen[v]:
                    stack.append(v)
        components.append(comp)
    if not components:
        return set()
    largest = max(components, key=len)
    return set(largest)


def filter_network_to_main_component(nodes_gdf, edges_gdf, verbose=True):
    """
    Keeps only nodes and links in the largest weak connected component.
    Drops nodes (and links touching them) that are disconnected from the main network.
    Returns (filtered_nodes_gdf, filtered_edges_gdf).
    """
    main_ids = get_main_component_node_ids(edges_gdf)
    nodes_filtered = nodes_gdf[nodes_gdf["node_id"].astype(int).isin(main_ids)].copy()
    edges_filtered = edges_gdf[
        edges_gdf["from_node_id"].astype(int).isin(main_ids)
        & edges_gdf["to_node_id"].astype(int).isin(main_ids)
    ].copy()
    if verbose:
        removed_nodes = len(nodes_gdf) - len(nodes_filtered)
        removed_edges = len(edges_gdf) - len(edges_filtered)
        if removed_nodes > 0 or removed_edges > 0:
            print(f"  Filtered to main component: removed {removed_nodes} disconnected nodes, {removed_edges} links.")
    return nodes_filtered, edges_filtered


def export_to_GIS(export_path): 
    # Make sure folder exists
    os.makedirs("Export", exist_ok=True)

    # Delete existing file if it exists
    if os.path.exists(export_path):
        os.remove(export_path)


def clean_gfts(stops_gdf, roads_filtered):
    stops_gdf = stops_gdf.dropna(subset=["geometry"])
    stops_gdf = stops_gdf[~stops_gdf.geometry.duplicated()]

    road_buffer = roads_filtered.buffer(50)
    stops_gdf = stops_gdf[stops_gdf.geometry.apply(
        lambda g: road_buffer.intersects(g).any()
    )]

    stops_gdf = stops_gdf.reset_index()
    stops_gdf = stops_gdf.set_index("stop_id")

    return stops_gdf


def load_ion_stops(csv_path, crs, stage1="LRT", stop_status="Constructed"):
    """
    Load ION LRT stops from ION_Stops.csv into a GeoDataFrame.
    Filters to Stage1 (e.g. LRT) and StopStatus (e.g. Constructed).
    X, Y are assumed to be in the same CRS as the road network (e.g. EPSG:26917).
    """
    import pandas as pd
    df = pd.read_csv(csv_path)
    df = df[(df["Stage1"] == stage1) & (df["StopStatus"] == stop_status)].copy()
    df = df.dropna(subset=["X", "Y"])
    geometry = gpd.points_from_xy(df["X"], df["Y"])
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs=crs)
    return gdf


def add_ion_stops_to_nodes(ion_stops_gdf, gmns_nodes, nodes, starting_node_id):
    """
    Add ION stops as new nodes at their exact coordinates (no snapping).
    Use this so LRT links connect real station locations and can have different travel times.
    """
    from shapely.geometry import Point
    node_id = starting_node_id
    for idx, row in ion_stops_gdf.iterrows():
        geom = row.geometry
        coord = (round(geom.x, 3), round(geom.y, 3))
        if coord in nodes:
            continue
        nodes[coord] = node_id
        gmns_nodes = pd.concat([
            gmns_nodes,
            gpd.GeoDataFrame([{
                "node_id": node_id,
                "x_coord": coord[0],
                "y_coord": coord[1],
                "node_type": "ion_stop",
                "geometry": Point(coord),
            }], geometry="geometry", crs=gmns_nodes.crs),
        ], ignore_index=True)
        node_id += 1
    return gmns_nodes, nodes, node_id


def snap_and_add_stops_to_nodes(stops_gdf, gmns_nodes, nodes, starting_node_id):
    """
    Snap stops to nearest road nodes and add them to GMNS nodes.
    
    Parameters:
    - stops_gdf: GeoDataFrame of filtered stops
    - gmns_nodes: GeoDataFrame of existing road nodes
    - nodes: dict mapping (x, y) -> node_id
    - starting_node_id: int, starting ID for new nodes
    
    Returns:
    - updated gmns_nodes GeoDataFrame
    - updated nodes dict
    - next available node_id
    """
    node_id = starting_node_id
    
    for stop_id, stop in stops_gdf.iterrows():
        # Snap to nearest road node
        distances = gmns_nodes.geometry.distance(stop.geometry)
        nearest_idx = distances.idxmin()
        nearest_geom = gmns_nodes.loc[nearest_idx, 'geometry']
        
        stops_gdf.at[stop_id, 'geometry'] = nearest_geom
        stops_gdf.at[stop_id, 'x'] = nearest_geom.x
        stops_gdf.at[stop_id, 'y'] = nearest_geom.y
        
        # Add to nodes if not already present
        coord = (nearest_geom.x, nearest_geom.y)
        if coord not in nodes:
            nodes[coord] = node_id
            gmns_nodes = pd.concat([
                gmns_nodes,
                gpd.GeoDataFrame([{
                    "node_id": node_id,
                    "x_coord": coord[0],
                    "y_coord": coord[1],
                    "node_type": "stop",
                    "geometry": Point(coord)
                }], geometry='geometry', crs=gmns_nodes.crs)
            ], ignore_index=True)
            node_id += 1
            
    return gmns_nodes, nodes, node_id




def shortest_path(project, orig_node, dest_node):

    # Get the car graph (mode must exist in allowed_uses!)
    graph = project.network.graphs['transit']   # OR 'c' depending on your parameters.yml

    graph.set_graph(cost_field='distance')
    graph.set_blocked_centroid_flows(False)

    # Shortest path
    graph.shortest_path(orig_node, dest_node)

    # Results
    path_nodes = graph.path_nodes # ordered list of nodes along shortest path
    path_links = graph.path_links # ordered list of link ids used in shortest path
    path_cost = graph.path_cost # total distance from o to d

    links = project.network.links.data
    path_links_gdf = links[links.link_id.isin(graph.path_links)]
    path_links_gdf.to_file("shortest_path.gpkg", layer="path", driver="GPKG")

    print("Path nodes: ", path_nodes)
    print("Path links: ", path_links)
    print("Path costs: ", path_cost)








# "10 15 3 7", target = 17 → True (10 + 7)
def has_pair_with_sum(num_string, target):
    num_list = list(map(int, num_string.split())) # 10 15 3 7 
    seen = set() # unordered collection with no duplicate elements / fast membership check 

    for n in num_list:
        if target - n in seen:
            return True
        seen.add(n)
    return False

num_string = "10 5 3 6 6"
target = 12
print(has_pair_with_sum(num_string, target))

# linked list -> order matters, need duplicates, need index
# sets -> order doesn't matter, just need to check if something exists in that list or not 
# dictionary -> maps keys to values, fast lookup 