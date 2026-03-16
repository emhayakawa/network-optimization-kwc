"""
Shortest path on the bus-only network by time, distance, or generalized cost.

Generalized Cost for Transit:
    GC = FARE + (WAITING_TIME + In-Vehicle Time) × VALUE_OF_TIME

Where:
    - FARE: flat transit fare (applied once per trip)
    - WAITING_TIME: average wait time at origin stop (applied once per trip)
    - In-Vehicle Time: travel time on bus (from GTFS)
    - VALUE_OF_TIME: time-to-money conversion ($/min)

Primary: AequilibraE (shortest_path_bus_aequilibrae) for compatibility with traffic assignment.
Fallback: Dijkstra (shortest_path_bus) when no project is available.
"""
import heapq
import pandas as pd

COST_TIME = "time"
COST_DISTANCE = "distance"
COST_GENERALIZED = "generalized"

# Default parameters (can be overridden in function calls)
DEFAULT_FARE = 3.50           # dollars
DEFAULT_WAITING_TIME = 7.5    # minutes
DEFAULT_VALUE_OF_TIME = 0.50  # dollars per minute


def _build_adjacency(links_df, cost_field):
    """Build adjacency list from links DataFrame."""
    adj = {}
    for _, row in links_df.iterrows():
        f, t = int(row["from_node_id"]), int(row["to_node_id"])
        c = float(row[cost_field])
        if f not in adj:
            adj[f] = []
        adj[f].append((t, c))
    return adj


def _build_adjacency_generalized(links_df, value_of_time):
    """
    Build adjacency list using generalized cost.
    
    Link cost = travel_time_min × value_of_time
    (Fare and waiting time are added once at the trip level, not per link)
    """
    adj = {}
    for _, row in links_df.iterrows():
        f, t = int(row["from_node_id"]), int(row["to_node_id"])
        travel_time = float(row["travel_time_min"])
        # Link cost is just the time component (fare/wait added at trip level)
        cost = travel_time * value_of_time
        if f not in adj:
            adj[f] = []
        adj[f].append((t, cost, travel_time))
    return adj


def shortest_path_bus(
    nodes_df, 
    links_df, 
    orig_node_id, 
    dest_node_id, 
    cost="time",
    fare=None,
    waiting_time_min=None,
    value_of_time=None,
    verbose=True
):
    """
    Compute shortest path on the bus network (Dijkstra).
    
    Parameters:
        nodes_df: Network nodes DataFrame
        links_df: Network links DataFrame
        orig_node_id: Origin node ID
        dest_node_id: Destination node ID
        cost: "time", "distance", or "generalized"
        fare: Transit fare in dollars (for generalized cost)
        waiting_time_min: Average waiting time at origin (for generalized cost)
        value_of_time: $/minute (for generalized cost)
        verbose: Print results
    
    Returns:
        dict: path_nodes, path_links, total_time_min, total_length_m, 
              generalized_cost (if applicable), found
    """
    # Set defaults
    if fare is None:
        fare = DEFAULT_FARE
    if waiting_time_min is None:
        waiting_time_min = DEFAULT_WAITING_TIME
    if value_of_time is None:
        value_of_time = DEFAULT_VALUE_OF_TIME
    
    # Build adjacency based on cost type
    if cost == COST_GENERALIZED:
        adj = _build_adjacency_generalized(links_df, value_of_time)
        use_generalized = True
    else:
        cost_field = "travel_time_min" if cost == COST_TIME else "length_m"
        adj = _build_adjacency(links_df, cost_field)
        use_generalized = False
    
    # Dijkstra's algorithm
    dist = {orig_node_id: 0.0}
    pred = {orig_node_id: None}
    pq = [(0.0, orig_node_id)]
    
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist.get(u, float("inf")):
            continue
        if u == dest_node_id:
            break
        
        for neighbor_data in adj.get(u, []):
            if use_generalized:
                v, c, _ = neighbor_data
            else:
                v, c = neighbor_data
            
            new_d = d + c
            if new_d < dist.get(v, float("inf")):
                dist[v] = new_d
                pred[v] = u
                heapq.heappush(pq, (new_d, v))
    
    # Check if path found
    if dest_node_id not in pred:
        if verbose:
            print(f"No path found from {orig_node_id} to {dest_node_id}")
        return {
            "path_nodes": None,
            "path_links": None,
            "total_time_min": None,
            "total_length_m": None,
            "generalized_cost": None,
            "found": False,
        }
    
    # Reconstruct path
    path_nodes = []
    u = dest_node_id
    while u is not None:
        path_nodes.append(u)
        u = pred[u]
    path_nodes.reverse()
    
    # Calculate totals
    path_links = [(path_nodes[i], path_nodes[i + 1]) for i in range(len(path_nodes) - 1)]
    link_lookup = links_df.set_index(["from_node_id", "to_node_id"])
    
    total_time_min = 0.0
    total_length_m = 0.0
    num_transfers = 0
    
    for (f, t) in path_links:
        try:
            rows = link_lookup.loc[(f, t)]
            row = rows.iloc[0] if isinstance(rows, pd.DataFrame) else rows
            total_time_min += float(row["travel_time_min"])
            total_length_m += float(row["length_m"])
            if row.get("link_type") == "transfer":
                num_transfers += 1
        except KeyError:
            continue
    
    # Calculate generalized cost
    # GC = FARE + (WAITING_TIME + In-Vehicle Time) × VALUE_OF_TIME
    total_travel_time = waiting_time_min + total_time_min
    generalized_cost = fare + (total_travel_time * value_of_time)
    
    if verbose:
        print(f"Shortest path ({cost}): {orig_node_id} -> {dest_node_id}")
        print(f"  In-vehicle time: {total_time_min:.2f} min")
        print(f"  Total distance: {total_length_m:.0f} m ({total_length_m/1000:.2f} km)")
        print(f"  Nodes: {len(path_nodes)}, Links: {len(path_links)}, Transfers: {num_transfers}")
        
        if cost == COST_GENERALIZED:
            print(f"\n  Generalized Cost Breakdown:")
            print(f"    Fare: ${fare:.2f}")
            print(f"    Waiting time: {waiting_time_min:.1f} min × ${value_of_time:.2f}/min = ${waiting_time_min * value_of_time:.2f}")
            print(f"    In-vehicle time: {total_time_min:.1f} min × ${value_of_time:.2f}/min = ${total_time_min * value_of_time:.2f}")
            print(f"    ─────────────────────────────────")
            print(f"    TOTAL GENERALIZED COST: ${generalized_cost:.2f}")
    
    return {
        "path_nodes": path_nodes,
        "path_links": path_links,
        "total_time_min": total_time_min,
        "total_length_m": total_length_m,
        "generalized_cost": generalized_cost,
        "waiting_time_min": waiting_time_min,
        "fare": fare,
        "num_transfers": num_transfers,
        "found": True,
    }


def compute_generalized_cost(
    total_time_min,
    fare=None,
    waiting_time_min=None,
    value_of_time=None
):
    """
    Compute generalized cost for a transit trip.
    
    GC = FARE + (WAITING_TIME + In-Vehicle Time) × VALUE_OF_TIME
    
    Parameters:
        total_time_min: In-vehicle travel time (minutes)
        fare: Transit fare ($)
        waiting_time_min: Wait time at origin (minutes)
        value_of_time: $/minute
    
    Returns:
        float: Generalized cost in dollars
    """
    if fare is None:
        fare = DEFAULT_FARE
    if waiting_time_min is None:
        waiting_time_min = DEFAULT_WAITING_TIME
    if value_of_time is None:
        value_of_time = DEFAULT_VALUE_OF_TIME
    
    total_travel_time = waiting_time_min + total_time_min
    return fare + (total_travel_time * value_of_time)


def shortest_path_bus_aequilibrae(
    project, orig_node_id, dest_node_id, cost="time", verbose=True
):
    """
    Compute shortest path using AequilibraE (same engine as traffic assignment).
    Returns dict: path_nodes, path_links (link IDs), total_time_min, total_length_m, found.
    
    Note: Generalized cost is computed post-hoc for AequilibraE paths.
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

    # Compute generalized cost
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


def node_ids_at_stop(nodes_df, stop_id):
    """Return list of node_ids for all (stop_id, route_id) at the given stop_id."""
    return nodes_df[nodes_df["stop_id"] == str(stop_id)]["node_id"].astype(int).tolist()


def check_connectivity(nodes_df, links_df, from_node_id, to_node_id, verbose=True):
    """Check whether two nodes exist and whether a path exists between them."""
    from collections import deque
    nids = set(nodes_df["node_id"].astype(int))
    from_ok = int(from_node_id) in nids
    to_ok = int(to_node_id) in nids
    if verbose:
        print("Connectivity check:")
        print(f"  Node {from_node_id} exists: {from_ok}")
        print(f"  Node {to_node_id} exists: {to_ok}")
    if not from_ok or not to_ok:
        return {"from_exists": from_ok, "to_exists": to_ok, "path_exists": False, "path_nodes": None}
    
    adj = {}
    for _, row in links_df.iterrows():
        f, t = int(row["from_node_id"]), int(row["to_node_id"])
        if f not in adj:
            adj[f] = []
        adj[f].append(t)
    
    q = deque([int(from_node_id)])
    pred = {int(from_node_id): None}
    while q:
        u = q.popleft()
        if u == int(to_node_id):
            break
        for v in adj.get(u, []):
            if v not in pred:
                pred[v] = u
                q.append(v)
    
    path_exists = int(to_node_id) in pred
    path_nodes = None
    if path_exists:
        path_nodes = []
        u = int(to_node_id)
        while u is not None:
            path_nodes.append(u)
            u = pred[u]
        path_nodes.reverse()
    
    if verbose:
        print(f"  Path exists: {path_exists}")
        if path_nodes:
            print(f"  Path length: {len(path_nodes)} nodes")
    
    return {"from_exists": from_ok, "to_exists": to_ok, "path_exists": path_exists, "path_nodes": path_nodes}


if __name__ == "__main__":
    import os
    import sys
    
    # Load pre-built network from CSV (avoids circular import)
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    nodes_path = os.path.join(data_dir, "nodes.csv")
    links_path = os.path.join(data_dir, "links.csv")
    
    if not os.path.exists(nodes_path) or not os.path.exists(links_path):
        print("ERROR: Network data not found. Run build_bus_network.py first.")
        print(f"  Expected: {nodes_path}")
        print(f"  Expected: {links_path}")
        sys.exit(1)
    
    print("=== Bus Network Shortest Path with Generalized Cost ===\n")
    
    # Load network from CSV
    print("Loading network from CSV...")
    nodes_df = pd.read_csv(nodes_path)
    links_df = pd.read_csv(links_path)
    print(f"  Nodes: {len(nodes_df)}, Links: {len(links_df)}")
    
    # Pick sample origin/destination
    all_node_ids = list(nodes_df["node_id"].values)
    orig = int(all_node_ids[0])
    dest = int(all_node_ids[min(100, len(all_node_ids) - 1)])
    
    print(f"\n=== Test: Node {orig} -> Node {dest} ===")
    
    # By time only
    print("\n--- Cost = TIME ---")
    result_time = shortest_path_bus(nodes_df, links_df, orig, dest, cost=COST_TIME)
    
    # By generalized cost
    print("\n--- Cost = GENERALIZED ---")
    result_gc = shortest_path_bus(
        nodes_df, links_df, orig, dest, 
        cost=COST_GENERALIZED,
        fare=DEFAULT_FARE,
        waiting_time_min=DEFAULT_WAITING_TIME,
        value_of_time=DEFAULT_VALUE_OF_TIME
    )
