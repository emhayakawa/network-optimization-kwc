"""
Bus network shortest path: AequilibraE for assignment compatibility.
Uses shared transit.shortest_path for Dijkstra and compute_generalized_cost.
"""
import os
import sys
import pandas as pd

# Ensure URA root is in path
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

from transit.shortest_path import (
    shortest_path_transit,
    compute_generalized_cost,
    node_ids_at_stop,
    check_connectivity,
    COST_TIME,
    COST_DISTANCE,
    COST_GENERALIZED,
    DEFAULT_FARE,
    DEFAULT_WAITING_TIME,
    DEFAULT_VALUE_OF_TIME,
)

# Backward compatibility alias
shortest_path_bus = shortest_path_transit


def shortest_path_bus_aequilibrae(
    project, orig_node_id, dest_node_id, cost="time", verbose=True
):
    """
    Compute shortest path using AequilibraE (same engine as traffic assignment).
    Returns dict: path_nodes, path_links (link IDs), total_time_min, total_length_m, found.
    Generalized cost is computed post-hoc.
    """
    graph_modes = list(project.network.graphs.keys())
    if not graph_modes:
        raise RuntimeError("No graphs built. Run project.network.build_graphs() first.")

    graph_key = "t" if "t" in graph_modes else ("c" if "c" in graph_modes else graph_modes[0])
    graph = project.network.graphs[graph_key]
    if verbose:
        print(f"  Graph mode: '{graph_key}' (available: {graph_modes})")

    links_data = project.network.links.data
    if cost == COST_TIME or cost == COST_GENERALIZED:
        cost_field = "travel_time"
    else:
        cost_field = "length"
    if cost_field == "length" and "length" not in links_data.columns and "distance" in links_data.columns:
        cost_field = "distance"
    graph.set_graph(cost_field)
    graph.set_blocked_centroid_flows(False)

    res = graph.compute_path(int(orig_node_id), int(dest_node_id))

    if res.path is None or len(res.path) == 0:
        if verbose:
            print("No path found from", orig_node_id, "to", dest_node_id)
        return {
            "path_nodes": None,
            "path_links": None,
            "total_time_min": None,
            "total_length_m": None,
            "generalized_cost": None,
            "found": False,
        }

    path_nodes = list(res.path_nodes)
    path_links = list(res.path)
    link_ids = [int(lid) for lid in path_links]
    path_links_df = links_data[links_data["link_id"].isin(link_ids)]
    length_col = "length" if "length" in links_data.columns else "distance"

    time_col = "travel_time_ab" if "travel_time_ab" in path_links_df.columns else "travel_time"
    total_time_min = path_links_df[time_col].sum() if time_col in path_links_df.columns else None
    total_length_m = path_links_df[length_col].sum() if length_col in path_links_df.columns else None

    if cost == COST_TIME and res.milepost is not None and len(res.milepost) > 0:
        total_time_min = float(res.milepost[-1])
    elif total_time_min is None or (isinstance(total_time_min, (int, float)) and pd.isna(total_time_min)):
        total_time_min = float(res.milepost[-1]) if cost == COST_TIME and res.milepost else 0.0
    if total_length_m is None or (isinstance(total_length_m, (int, float)) and pd.isna(total_length_m)):
        total_length_m = float(res.milepost[-1]) if cost == COST_DISTANCE and res.milepost else 0.0

    generalized_cost = compute_generalized_cost(total_time_min)

    if verbose:
        print(f"Shortest path ({cost}) [AequilibraE]: {orig_node_id} -> {dest_node_id}")
        print(f"  Total time: {total_time_min:.2f} min")
        print(f"  Total distance: {total_length_m:.2f} m")
        print(f"  Path nodes: {len(path_nodes)}, Links: {len(path_links)}")

        if cost == COST_GENERALIZED:
            print(f"\n  Generalized Cost: ${generalized_cost:.2f}")
            print(f"    (Fare: ${DEFAULT_FARE:.2f} + Wait: {DEFAULT_WAITING_TIME:.1f}min + Travel: {total_time_min:.1f}min) × ${DEFAULT_VALUE_OF_TIME:.2f}/min")

    return {
        "path_nodes": path_nodes,
        "path_links": path_links,
        "total_time_min": float(total_time_min),
        "total_length_m": float(total_length_m),
        "generalized_cost": generalized_cost,
        "found": True,
    }
