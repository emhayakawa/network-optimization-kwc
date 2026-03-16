"""
Node creation and processing functions for the road network.
Nodes are created from:
1. Road intersections (endpoints appearing in 2+ roads)
2. Traffic lights (snapped to nearby intersections or added as new nodes)

Includes clustering to merge nearby nodes (e.g., divided road intersections).
"""
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point
from collections import defaultdict
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist

from .config import SNAP_TOLERANCE_M


def snap(coord, precision=3):
    """Round coordinate to given precision for consistent node matching."""
    return (round(coord[0], precision), round(coord[1], precision))


def create_nodes_from_roads(roads_gdf):
    """
    Create nodes from road intersections (endpoints that appear in multiple roads).
    
    Returns:
        nodes_data: list of dicts with node_id, x_coord, y_coord, geometry, signalized
        nodes: dict mapping (x, y) -> node_id
        next_node_id: next available node ID
    """
    coord_count = defaultdict(int)
    
    for idx, row in roads_gdf.iterrows():
        coords = list(row.geometry.coords)
        start, end = snap(coords[0]), snap(coords[-1])
        coord_count[start] += 1
        coord_count[end] += 1
    
    nodes = {}
    node_id = 1
    
    for coord, count in coord_count.items():
        if count >= 2:  # intersections only
            nodes[coord] = node_id
            node_id += 1
    
    nodes_data = [
        {
            'node_id': nid,
            'x_coord': coord[0],
            'y_coord': coord[1],
            'node_type': 'intersection',
            'signalized': 0,
            'geometry': Point(coord)
        }
        for coord, nid in nodes.items()
    ]
    
    return nodes_data, nodes, node_id


def cluster_nearby_nodes(nodes_gdf, cluster_tolerance_m):
    """
    Cluster nodes that are within cluster_tolerance_m of each other.
    This handles divided roads where one intersection creates multiple nodes.
    
    Parameters:
        nodes_gdf: GeoDataFrame of nodes
        cluster_tolerance_m: distance threshold for clustering (e.g., 50m)
    
    Returns:
        nodes_gdf with 'cluster_id' column, mapping dict from old coords to new coords
    """
    if len(nodes_gdf) < 2:
        nodes_gdf = nodes_gdf.copy()
        nodes_gdf['cluster_id'] = range(len(nodes_gdf))
        return nodes_gdf, {}
    
    coords = np.array([[g.x, g.y] for g in nodes_gdf.geometry])
    
    # Compute pairwise distances and cluster
    if len(coords) > 1:
        distances = pdist(coords)
        if len(distances) > 0 and np.max(distances) > 0:
            Z = linkage(distances, method='complete')
            cluster_ids = fcluster(Z, t=cluster_tolerance_m, criterion='distance')
        else:
            cluster_ids = np.arange(1, len(coords) + 1)
    else:
        cluster_ids = np.array([1])
    
    nodes_gdf = nodes_gdf.copy()
    nodes_gdf['cluster_id'] = cluster_ids
    
    return nodes_gdf, cluster_ids


def merge_clustered_nodes(nodes_gdf, nodes_dict):
    """
    Merge nodes in the same cluster into a single node at the centroid.
    Updates nodes_dict to map old coordinates to new representative coordinates.
    
    Parameters:
        nodes_gdf: GeoDataFrame with 'cluster_id' column
        nodes_dict: dict mapping (x, y) -> node_id
    
    Returns:
        merged_nodes_gdf: GeoDataFrame with one node per cluster
        new_nodes_dict: updated mapping from ALL old coords to new node_ids
        coord_remap: dict mapping old (x,y) -> new (x,y) for updating edges
    """
    merged_nodes = []
    coord_remap = {}  # old coord -> new coord
    new_nodes_dict = {}
    next_node_id = 1
    
    for cluster_id, group in nodes_gdf.groupby('cluster_id'):
        # Compute centroid of cluster
        centroid_x = group['x_coord'].mean()
        centroid_y = group['y_coord'].mean()
        new_coord = snap((centroid_x, centroid_y))
        
        # Check if any node in cluster is signalized
        is_signalized = 1 if (group['signalized'] == 1).any() else 0
        
        # Determine node type
        if is_signalized:
            node_type = 'signalized_intersection'
        elif len(group) > 1:
            node_type = 'merged_intersection'
        else:
            node_type = group.iloc[0]['node_type']
        
        merged_nodes.append({
            'node_id': next_node_id,
            'x_coord': new_coord[0],
            'y_coord': new_coord[1],
            'node_type': node_type,
            'signalized': is_signalized,
            'geometry': Point(new_coord),
            'merged_count': len(group),
        })
        
        # Map all old coordinates in this cluster to the new coordinate
        for _, row in group.iterrows():
            old_coord = snap((row['x_coord'], row['y_coord']))
            coord_remap[old_coord] = new_coord
        
        new_nodes_dict[new_coord] = next_node_id
        next_node_id += 1
    
    merged_nodes_gdf = gpd.GeoDataFrame(merged_nodes, geometry='geometry', crs=nodes_gdf.crs)
    
    return merged_nodes_gdf, new_nodes_dict, coord_remap


def load_traffic_lights(shapefile_path, target_crs):
    """
    Load traffic lights from shapefile and reproject to target CRS.
    
    Returns:
        GeoDataFrame of traffic lights with geometry in target_crs
    """
    lights_gdf = gpd.read_file(shapefile_path)
    
    # Filter to active signals only if STATUS column exists
    if 'STATUS' in lights_gdf.columns:
        lights_gdf = lights_gdf[lights_gdf['STATUS'] == 'ACTIVE'].copy()
    
    # Reproject to target CRS
    if lights_gdf.crs != target_crs:
        lights_gdf = lights_gdf.to_crs(target_crs)
    
    return lights_gdf


def add_traffic_lights_to_nodes(lights_gdf, nodes_gdf, nodes_dict, next_node_id, snap_tolerance=None):
    """
    Add traffic lights to the network:
    - If a light is within snap_tolerance of an existing node, mark that node as signalized
    - Otherwise, add the light as a new signalized node
    
    Parameters:
        lights_gdf: GeoDataFrame of traffic lights
        nodes_gdf: GeoDataFrame of existing road nodes
        nodes_dict: dict mapping (x, y) -> node_id
        next_node_id: next available node ID
        snap_tolerance: distance in meters to snap lights to existing nodes
    
    Returns:
        updated nodes_gdf, nodes_dict, next_node_id
    """
    if snap_tolerance is None:
        snap_tolerance = SNAP_TOLERANCE_M
    
    nodes_gdf = nodes_gdf.copy()
    signalized_node_ids = set()
    new_nodes = []
    
    for idx, light in lights_gdf.iterrows():
        light_point = light.geometry
        
        # Find nearest existing node
        distances = nodes_gdf.geometry.distance(light_point)
        min_dist = distances.min()
        nearest_idx = distances.idxmin()
        
        if min_dist <= snap_tolerance:
            # Snap to existing node - mark it as signalized
            node_id = nodes_gdf.loc[nearest_idx, 'node_id']
            signalized_node_ids.add(node_id)
        else:
            # Add as new node
            coord = snap((light_point.x, light_point.y))
            
            if coord in nodes_dict:
                signalized_node_ids.add(nodes_dict[coord])
            else:
                nodes_dict[coord] = next_node_id
                new_nodes.append({
                    'node_id': next_node_id,
                    'x_coord': coord[0],
                    'y_coord': coord[1],
                    'node_type': 'traffic_light',
                    'signalized': 1,
                    'geometry': Point(coord)
                })
                next_node_id += 1
    
    # Update signalized field for snapped nodes
    nodes_gdf.loc[nodes_gdf['node_id'].isin(signalized_node_ids), 'signalized'] = 1
    
    # Add new traffic light nodes
    if new_nodes:
        new_nodes_gdf = gpd.GeoDataFrame(new_nodes, geometry='geometry', crs=nodes_gdf.crs)
        nodes_gdf = pd.concat([nodes_gdf, new_nodes_gdf], ignore_index=True)
    
    return nodes_gdf, nodes_dict, next_node_id


def get_main_component_node_ids(edges_df):
    """
    Return node_ids in the largest weakly connected component.
    Uses BFS on undirected graph representation.
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
    return set(max(components, key=len))


def filter_to_main_component(nodes_gdf, edges_gdf, verbose=True):
    """
    Keep only nodes and links in the largest connected component.
    
    Returns:
        filtered_nodes_gdf, filtered_edges_gdf
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
            print(f"  Removed {removed_nodes} disconnected nodes, {removed_edges} links")
    
    return nodes_filtered, edges_filtered
