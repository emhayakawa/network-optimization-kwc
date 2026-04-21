"""
Shortest path computation with generalized cost function.

Supports:
- NetworkX (flexible, custom costs)
- AequilibraE (fast, limited to built-in fields)

Generalized Cost = (distance_km * COST_PER_KM) + (travel_time_hours * VALUE_OF_TIME)
"""
import os
import networkx as nx
import pandas as pd
import geopandas as gpd

from .config import COST_PER_KM, VALUE_OF_TIME, SIGNAL_DELAY_SECONDS, ARCGIS_EXPORT_DIR


def compute_generalized_cost(edges_gdf, nodes_gdf=None):
    """
    Compute generalized cost for each link.
    
    Generalized Cost = (distance_km * COST_PER_KM) + (travel_time_hours * VALUE_OF_TIME)
    """
    length_km = edges_gdf['length'] / 1000.0
    travel_time_hours = edges_gdf['travel_time_min'] / 60.0
    
    distance_cost = length_km * COST_PER_KM
    time_cost = travel_time_hours * VALUE_OF_TIME
    
    return distance_cost + time_cost


def add_generalized_cost_to_edges(edges_gdf, nodes_gdf=None):
    """Add 'generalized_cost' column to edges GeoDataFrame."""
    edges_gdf = edges_gdf.copy()
    edges_gdf['generalized_cost'] = compute_generalized_cost(edges_gdf, nodes_gdf)
    return edges_gdf


def build_networkx_graph(edges_gdf, weight_field='generalized_cost'):
    """
    Build a NetworkX DiGraph from edges GeoDataFrame.
    
    Parameters:
        edges_gdf: GeoDataFrame with from_node_id, to_node_id, and weight columns
        weight_field: column to use as edge weight (default: 'generalized_cost')
    
    Returns:
        nx.DiGraph with edge weights
    """
    G = nx.DiGraph()
    
    for _, row in edges_gdf.iterrows():
        G.add_edge(
            int(row['from_node_id']),
            int(row['to_node_id']),
            weight=row[weight_field],
            link_id=int(row['link_id']),
            length=row['length'],
            travel_time_min=row.get('travel_time_min', 0),
            generalized_cost=row.get('generalized_cost', 0),
        )
    
    return G


def _path_nodes_to_link_ids(G, path_nodes):
    """Map a node sequence to link_id values using edges on ``G``."""
    path_links = []
    for i in range(len(path_nodes) - 1):
        edge_data = G.get_edge_data(path_nodes[i], path_nodes[i + 1])
        if edge_data:
            path_links.append(edge_data['link_id'])
    return path_links


def compute_shortest_path_networkx_from_graph(G, orig_node, dest_node, verbose=False):
    """
    Shortest path on a pre-built DiGraph using the ``weight`` edge attribute.

    Returns the same dict shape as :func:`compute_shortest_path_networkx`.
    """
    try:
        path_nodes = nx.dijkstra_path(G, orig_node, dest_node, weight='weight')
        total_cost = nx.dijkstra_path_length(G, orig_node, dest_node, weight='weight')
        path_links = _path_nodes_to_link_ids(G, path_nodes)
        if verbose:
            print(f"  Path found! Total cost: {total_cost:.4f}")
        return {
            "path_nodes": path_nodes,
            "path_links": path_links,
            "total_cost": total_cost,
            "found": True,
        }
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        if verbose:
            print("  No path found!")
        return {
            "path_nodes": None,
            "path_links": None,
            "total_cost": None,
            "found": False,
        }


def shortest_path_zone_sets_multi_source(G, orig_nodes, dest_nodes):
    """
    Best path between two *sets* of nodes (e.g. all nodes in origin TAZ vs destination TAZ).

    Uses one multi-source Dijkstra from all ``orig_nodes`` (distance 0 at each source), then
    picks the destination node with minimum cost and recovers one shortest path.

    Parameters:
        G: nx.DiGraph from :func:`build_networkx_graph`
        orig_nodes: non-empty list of origin node IDs
        dest_nodes: non-empty list of destination node IDs

    Returns:
        Same dict keys as :func:`compute_shortest_path_networkx`, plus ``orig_node_id`` and
        ``dest_node_id`` when ``found`` is True.
    """
    empty = {
        "path_nodes": None,
        "path_links": None,
        "total_cost": None,
        "found": False,
        "orig_node_id": None,
        "dest_node_id": None,
    }
    if not orig_nodes or not dest_nodes:
        return empty

    inf = float("inf")
    dists = nx.multi_source_dijkstra_path_length(G, orig_nodes, weight="weight")
    best_dest = None
    best_cost = inf
    for d in dest_nodes:
        c = dists.get(d, inf)
        if c < best_cost:
            best_cost = c
            best_dest = d

    if best_dest is None or best_cost == inf:
        return empty

    try:
        total_cost, path_nodes = nx.multi_source_dijkstra(
            G, orig_nodes, target=best_dest, weight="weight"
        )
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return empty

    path_links = _path_nodes_to_link_ids(G, path_nodes)
    return {
        "path_nodes": path_nodes,
        "path_links": path_links,
        "total_cost": float(total_cost),
        "found": True,
        "orig_node_id": int(path_nodes[0]),
        "dest_node_id": int(path_nodes[-1]),
    }


def compute_shortest_path_networkx(edges_gdf, orig_node, dest_node, 
                                    weight_field='generalized_cost', verbose=True):
    """
    Compute shortest path using NetworkX with custom cost field.
    
    Parameters:
        edges_gdf: GeoDataFrame with edges (must have weight_field column)
        orig_node: origin node ID
        dest_node: destination node ID
        weight_field: column to minimize ('generalized_cost', 'length', 'travel_time_min')
        verbose: print output
    
    Returns:
        dict with path_nodes, path_links, total_cost, found
    """
    if verbose:
        print(f"Computing shortest path (NetworkX): {orig_node} -> {dest_node}")
        print(f"  Cost field: {weight_field}")
    
    G = build_networkx_graph(edges_gdf, weight_field)
    return compute_shortest_path_networkx_from_graph(G, orig_node, dest_node, verbose=verbose)


def compute_shortest_path(project, orig_node, dest_node, cost_field="length", verbose=True):
    """
    Compute shortest path using AequilibraE (limited to built-in fields like 'length').
    
    For custom cost fields like 'generalized_cost', use compute_shortest_path_networkx() instead.
    """
    if verbose:
        print(f"Computing shortest path (AequilibraE): {orig_node} -> {dest_node}")
        print(f"  Cost field: {cost_field}")
    
    graph_modes = list(project.network.graphs.keys())
    if not graph_modes:
        raise RuntimeError("No graphs built. Call project.network.build_graphs() first.")
    
    graph = project.network.graphs.get("c") or project.network.graphs[graph_modes[0]]
    
    graph.set_graph(cost_field)
    graph.set_blocked_centroid_flows(False)
    
    res = graph.compute_path(orig_node, dest_node)
    
    if res.path is not None and len(res.path) > 0:
        path_nodes = list(res.path_nodes)
        path_links = list(res.path)
        total_cost = res.milepost[-1] if len(res.milepost) > 0 else 0.0
        
        if verbose:
            if cost_field == "length":
                print(f"  Total cost: {total_cost/1000:.4f} km")
            else:
                print(f"  Total cost: {total_cost:.4f}")
            print(f"  Nodes: {len(path_nodes)}, Links: {len(path_links)}")
            print(f"  Node sequence (origin -> destination): {path_nodes}")
        
        return {
            "path_nodes": path_nodes,
            "path_links": path_links,
            "total_cost": total_cost,
            "found": True,
        }
    else:
        if verbose:
            print("  No path found!")
        return {
            "path_nodes": None,
            "path_links": None,
            "total_cost": None,
            "found": False,
        }


def compute_path_details(edges_gdf, path_links):
    """
    Get detailed cost breakdown for a path.
    
    Parameters:
        edges_gdf: GeoDataFrame of all edges (with generalized_cost column)
        path_links: list of link IDs from shortest path result
    
    Returns:
        dict with distance, time, and cost breakdowns
    """
    path_edges = edges_gdf[edges_gdf['link_id'].isin(path_links)].copy()
    
    total_distance_m = path_edges['length'].sum()
    total_distance_km = total_distance_m / 1000.0
    
    if 'travel_time_min' in path_edges.columns:
        total_travel_time_min = path_edges['travel_time_min'].sum()
    else:
        total_travel_time_min = (total_distance_km / 50) * 60
    
    total_travel_time_hours = total_travel_time_min / 60.0
    
    distance_cost = total_distance_km * COST_PER_KM
    time_cost = total_travel_time_hours * VALUE_OF_TIME
    generalized_cost = distance_cost + time_cost
    
    return {
        "total_distance_km": round(total_distance_km, 3),
        "total_travel_time_min": round(total_travel_time_min, 2),
        "total_distance_cost": round(distance_cost, 2),
        "total_time_cost": round(time_cost, 2),
        "total_generalized_cost": round(generalized_cost, 2),
        "path_edges": path_edges,
    }


def export_path_to_gis(edges_gdf, path_links, output_path):
    """Export a shortest path to GeoPackage for visualization."""
    path_gdf = edges_gdf[edges_gdf['link_id'].isin(path_links)]
    path_gdf.to_file(output_path, layer="path", driver="GPKG")
    print(f"Path exported to: {output_path}")


def export_shortest_path_to_arcgis(nodes_gdf, edges_gdf, path_nodes, path_links, 
                                    output_path=None, verbose=True):
    """
    Export shortest path nodes and links to GeoPackage for ArcGIS/QGIS visualization.
    
    Parameters:
        nodes_gdf: GeoDataFrame of all nodes
        edges_gdf: GeoDataFrame of all edges
        path_nodes: list of node IDs in order from origin to destination
        path_links: list of link IDs in the path
        output_path: path to output GeoPackage file (default: ARCGIS_EXPORT_DIR/shortest_path.gpkg)
        verbose: print progress
    
    Returns:
        str: path to the exported GeoPackage file
    """
    if output_path is None:
        os.makedirs(ARCGIS_EXPORT_DIR, exist_ok=True)
        output_path = os.path.join(ARCGIS_EXPORT_DIR, "shortest_path.gpkg")
    
    # Remove existing file to ensure clean export
    if os.path.exists(output_path):
        os.remove(output_path)
    
    # Filter nodes in the path and preserve order
    path_nodes_gdf = nodes_gdf[nodes_gdf['node_id'].isin(path_nodes)].copy()
    path_nodes_gdf['path_order'] = path_nodes_gdf['node_id'].apply(lambda x: path_nodes.index(x))
    path_nodes_gdf = path_nodes_gdf.sort_values('path_order').reset_index(drop=True)
    
    # Mark origin and destination
    path_nodes_gdf['node_type'] = 'intermediate'
    path_nodes_gdf.loc[path_nodes_gdf['path_order'] == 0, 'node_type'] = 'origin'
    path_nodes_gdf.loc[path_nodes_gdf['path_order'] == len(path_nodes) - 1, 'node_type'] = 'destination'
    
    # Filter links in the path and preserve order
    path_links_gdf = edges_gdf[edges_gdf['link_id'].isin(path_links)].copy()
    path_links_gdf['path_order'] = path_links_gdf['link_id'].apply(lambda x: path_links.index(x))
    path_links_gdf = path_links_gdf.sort_values('path_order').reset_index(drop=True)
    
    # Convert to WGS84 for ArcGIS compatibility
    path_nodes_wgs = path_nodes_gdf.to_crs("EPSG:4326")
    path_links_wgs = path_links_gdf.to_crs("EPSG:4326")
    
    # Ensure ID columns are integers
    path_nodes_wgs['node_id'] = path_nodes_wgs['node_id'].astype(int)
    path_links_wgs['link_id'] = path_links_wgs['link_id'].astype(int)
    path_links_wgs['from_node_id'] = path_links_wgs['from_node_id'].astype(int)
    path_links_wgs['to_node_id'] = path_links_wgs['to_node_id'].astype(int)
    
    # Export to GeoPackage with nodes and links layers
    path_nodes_wgs.to_file(output_path, layer="nodes", driver="GPKG")
    path_links_wgs.to_file(output_path, layer="links", driver="GPKG", mode="a")
    
    if verbose:
        print(f"\nShortest path exported to: {output_path}")
        print(f"  Nodes layer: {len(path_nodes_wgs)} nodes (with path_order and node_type fields)")
        print(f"  Links layer: {len(path_links_wgs)} links (with path_order field)")
        print(f"  Origin node: {path_nodes[0]}, Destination node: {path_nodes[-1]}")
    
    return output_path
