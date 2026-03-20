"""
Shortest path on transit networks (bus-only or multimodal ION) by time, distance, or generalized cost.

Generalized Cost for Transit:
    GC = FARE + (WAITING_TIME + In-Vehicle Time) × VALUE_OF_TIME

Where:
    - FARE: flat transit fare (applied once per trip)
    - WAITING_TIME: average wait time at origin stop (applied once per trip)
    - In-Vehicle Time: travel time (from GTFS)
    - VALUE_OF_TIME: time-to-money conversion ($/min)

Used by both bus_network and ion_network for consistent computation.
"""
import heapq
import os
import pandas as pd

COST_TIME = "time"
COST_DISTANCE = "distance"
COST_GENERALIZED = "generalized"

# Default parameters (can be overridden in function calls)
DEFAULT_FARE = 3.50           # dollars (GRT unified fare)
DEFAULT_WAITING_TIME = 7.5    # minutes
DEFAULT_VALUE_OF_TIME = 0.33  # dollars per minute ($20/hour)


def _get_dest_cluster_nodes(nodes_df, dest_node_id):
    """Return set of node_ids in the same cluster as dest_node_id. Used to waive transfer cost when arriving at destination cluster."""
    dest_node_id = int(dest_node_id)
    if "cluster_id" not in nodes_df.columns:
        return {dest_node_id}
    dest_row = nodes_df[nodes_df["node_id"] == dest_node_id]
    if len(dest_row) == 0:
        return {dest_node_id}
    dest_cluster = dest_row["cluster_id"].iloc[0]
    same_cluster = nodes_df[nodes_df["cluster_id"] == dest_cluster]["node_id"].astype(int).tolist()
    return set(same_cluster)


def _build_adjacency(links_df, cost_field, dest_cluster_nodes=None):
    """Build adjacency list from links DataFrame."""
    adj = {}
    dest_cluster_nodes = dest_cluster_nodes or set()
    for _, row in links_df.iterrows():
        f, t = int(row["from_node_id"]), int(row["to_node_id"])
        link_type = row.get("link_type", "")
        if link_type in ("transfer", "multimodal_transfer") and t in dest_cluster_nodes:
            c = 0.0
        else:
            c = float(row[cost_field])
        if f not in adj:
            adj[f] = []
        adj[f].append((t, c))
    return adj


def _build_adjacency_generalized(links_df, value_of_time, dest_cluster_nodes=None):
    """
    Build adjacency list using generalized cost.
    Link cost = travel_time_min × value_of_time
    Transfer/multimodal_transfer links to destination cluster cost 0 (arrival at destination).
    """
    adj = {}
    dest_cluster_nodes = dest_cluster_nodes or set()
    for _, row in links_df.iterrows():
        f, t = int(row["from_node_id"]), int(row["to_node_id"])
        link_type = row.get("link_type", "")
        if link_type in ("transfer", "multimodal_transfer") and t in dest_cluster_nodes:
            travel_time = 0.0
            cost = 0.0
        else:
            travel_time = float(row["travel_time_min"])
            cost = travel_time * value_of_time
        if f not in adj:
            adj[f] = []
        adj[f].append((t, cost, travel_time))
    return adj


def shortest_path_transit(
    nodes_df,
    links_df,
    orig_node_id,
    dest_node_id,
    cost="generalized",
    fare=None,
    waiting_time_min=None,
    value_of_time=None,
    verbose=True,
):
    """
    Compute shortest path on a transit network (Dijkstra).
    Works for bus-only or multimodal (bus + LRT) networks.

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
              generalized_cost, num_transfers, modes_used, routes_used, found
    """
    if fare is None:
        fare = DEFAULT_FARE
    if waiting_time_min is None:
        waiting_time_min = DEFAULT_WAITING_TIME
    if value_of_time is None:
        value_of_time = DEFAULT_VALUE_OF_TIME

    orig_node_id = int(orig_node_id)
    dest_node_id = int(dest_node_id)
    dest_cluster_nodes = _get_dest_cluster_nodes(nodes_df, dest_node_id)

    if cost == COST_GENERALIZED:
        adj = _build_adjacency_generalized(links_df, value_of_time, dest_cluster_nodes)
        use_generalized = True
    else:
        cost_field = "travel_time_min" if cost == COST_TIME else "length_m"
        adj = _build_adjacency(links_df, cost_field, dest_cluster_nodes)
        use_generalized = False

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

    if dest_node_id not in pred:
        if verbose:
            print(f"No path found from {orig_node_id} to {dest_node_id}")
        return {
            "path_nodes": None,
            "path_links": None,
            "total_time_min": None,
            "total_length_m": None,
            "generalized_cost": None,
            "num_transfers": None,
            "num_multimodal_transfers": None,
            "modes_used": None,
            "routes_used": None,
            "found": False,
        }

    path_nodes = []
    u = dest_node_id
    while u is not None:
        path_nodes.append(u)
        u = pred[u]
    path_nodes.reverse()

    path_links = [(path_nodes[i], path_nodes[i + 1]) for i in range(len(path_nodes) - 1)]
    link_lookup = links_df.set_index(["from_node_id", "to_node_id"])

    total_time_min = 0.0
    total_length_m = 0.0
    num_transfers = 0
    num_multimodal_transfers = 0
    modes_used = set()
    routes_used = []

    for (f, t) in path_links:
        try:
            rows = link_lookup.loc[(f, t)]
            row = rows.iloc[0] if isinstance(rows, pd.DataFrame) else rows
            link_type = row.get("link_type", "")
            if link_type in ("transfer", "multimodal_transfer") and t in dest_cluster_nodes:
                total_time_min += 0.0  # No transfer cost when arriving at destination cluster
            else:
                total_time_min += float(row["travel_time_min"])
            total_length_m += float(row["length_m"])
            mode = row.get("mode", "")

            if link_type == "transfer":
                num_transfers += 1
            elif link_type == "multimodal_transfer":
                num_transfers += 1
                num_multimodal_transfers += 1
            elif link_type == "route":
                if mode:
                    modes_used.add(mode)
                route_id = row.get("route_id")
                if route_id and (not routes_used or routes_used[-1] != str(route_id)):
                    routes_used.append(str(route_id))
        except KeyError:
            continue

    total_travel_time = waiting_time_min + total_time_min
    generalized_cost = fare + (total_travel_time * value_of_time)

    if verbose:
        print(f"\nShortest path ({cost}): {orig_node_id} -> {dest_node_id}")
        print(f"  In-vehicle time: {total_time_min:.2f} min")
        print(f"  Total distance: {total_length_m:.0f} m ({total_length_m/1000:.2f} km)")
        print(f"  Nodes: {len(path_nodes)}, Links: {len(path_links)}")
        print(f"  Node sequence (origin -> destination): {path_nodes}")
        print(f"  Transfers: {num_transfers}" + (f" (multimodal: {num_multimodal_transfers})" if num_multimodal_transfers else ""))
        if modes_used:
            print(f"  Modes used: {', '.join(sorted(modes_used))}")

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
        "num_multimodal_transfers": num_multimodal_transfers,
        "modes_used": list(modes_used),
        "routes_used": routes_used,
        "found": True,
    }


def compute_generalized_cost(
    total_time_min,
    fare=None,
    waiting_time_min=None,
    value_of_time=None,
):
    """Compute generalized cost: GC = FARE + (WAITING_TIME + In-Vehicle Time) × VALUE_OF_TIME"""
    if fare is None:
        fare = DEFAULT_FARE
    if waiting_time_min is None:
        waiting_time_min = DEFAULT_WAITING_TIME
    if value_of_time is None:
        value_of_time = DEFAULT_VALUE_OF_TIME

    total_travel_time = waiting_time_min + total_time_min
    return fare + (total_travel_time * value_of_time)


def compute_path_details(links_df, path_links, nodes_df=None):
    """
    Compute detailed path statistics.
    Handles both bus (no mode column) and multimodal (mode column) links.
    When nodes_df provided, transfer links to destination cluster count as 0 time.
    """
    if not path_links:
        return {
            "total_length_m": 0,
            "total_travel_time_min": 0,
            "num_transfers": 0,
            "modes_used": [],
            "routes_used": [],
            "segments_by_mode": {"bus": 0, "lrt": 0, "transfer": 0},
        }

    dest_node = path_links[-1][1]  # path ends at destination
    dest_cluster_nodes = _get_dest_cluster_nodes(nodes_df, dest_node) if nodes_df is not None else set()

    link_lookup = links_df.set_index(["from_node_id", "to_node_id"])
    total_length_m = 0.0
    total_travel_time_min = 0.0
    num_transfers = 0
    modes_used = set()
    routes_used = []
    segments_by_mode = {"bus": 0, "lrt": 0, "transfer": 0}

    for (f, t) in path_links:
        try:
            rows = link_lookup.loc[(f, t)]
            row = rows.iloc[0] if isinstance(rows, pd.DataFrame) else rows
            link_type = row.get("link_type", "")
            if link_type in ("transfer", "multimodal_transfer") and t in dest_cluster_nodes:
                total_travel_time_min += 0.0
            else:
                total_travel_time_min += float(row["travel_time_min"])
            total_length_m += float(row["length_m"])
            mode = row.get("mode", "")

            if "transfer" in link_type:
                num_transfers += 1
                segments_by_mode["transfer"] += 1
            elif link_type == "route":
                if mode == "bus":
                    segments_by_mode["bus"] += 1
                elif mode == "lrt":
                    segments_by_mode["lrt"] += 1
                if mode:
                    modes_used.add(mode)
                route_id = row.get("route_id")
                if route_id and (not routes_used or routes_used[-1] != str(route_id)):
                    routes_used.append(str(route_id))
        except KeyError:
            continue

    return {
        "total_length_m": total_length_m,
        "total_travel_time_min": total_travel_time_min,
        "num_transfers": num_transfers,
        "modes_used": list(modes_used),
        "routes_used": routes_used,
        "segments_by_mode": segments_by_mode,
    }


def node_ids_at_stop(nodes_df, stop_id):
    """Return list of node_ids for all nodes at the given stop_id."""
    return nodes_df[nodes_df["stop_id"] == str(stop_id)]["node_id"].astype(int).tolist()


def node_ids_by_mode(nodes_df, mode):
    """Return list of node_ids for nodes with the given mode. Requires 'mode' column in nodes."""
    if "mode" not in nodes_df.columns:
        return []
    return nodes_df[nodes_df["mode"] == mode]["node_id"].astype(int).tolist()


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

    return {
        "from_exists": from_ok,
        "to_exists": to_ok,
        "path_exists": path_exists,
        "path_nodes": path_nodes,
    }


def export_shortest_path_to_arcgis(nodes_df, links_df, path_nodes, path_links, out_dir=None, link_shapes=None):
    """
    Export the shortest path to a GeoPackage for ArcGIS visualization.
    """
    import geopandas as gpd
    from shapely.geometry import LineString, Point

    if out_dir is None:
        out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "arcgis_export")
    os.makedirs(out_dir, exist_ok=True)

    if link_shapes is None:
        link_shapes = {}

    gpkg_path = os.path.join(out_dir, "shortest_path.gpkg")

    path_nodes_df = nodes_df[nodes_df["node_id"].isin(path_nodes)].copy()
    gdf_path_nodes = gpd.GeoDataFrame(
        path_nodes_df,
        geometry=[Point(row["stop_lon"], row["stop_lat"]) for _, row in path_nodes_df.iterrows()],
        crs="EPSG:4326",
    )

    node_coord = nodes_df.set_index("node_id")[["stop_lat", "stop_lon"]].to_dict("index")
    link_lookup = links_df.set_index(["from_node_id", "to_node_id"])

    path_links_data = []
    n_with_shapes = 0

    for (f, t) in path_links:
        try:
            rows = link_lookup.loc[(f, t)]
            row = rows.iloc[0] if isinstance(rows, pd.DataFrame) else rows
            c1, c2 = node_coord.get(f), node_coord.get(t)

            if c1 and c2:
                shape_coords = link_shapes.get((f, t))
                if shape_coords and len(shape_coords) >= 2:
                    geom = LineString(shape_coords)
                    n_with_shapes += 1
                else:
                    geom = LineString([(c1["stop_lon"], c1["stop_lat"]), (c2["stop_lon"], c2["stop_lat"])])

                path_links_data.append({
                    "from_node_id": f,
                    "to_node_id": t,
                    "link_type": row.get("link_type", ""),
                    "route_id": row.get("route_id", ""),
                    "mode": row.get("mode", ""),
                    "travel_time_min": row.get("travel_time_min", 0),
                    "length_m": row.get("length_m", 0),
                    "geometry": geom,
                })
        except KeyError:
            continue

    if path_links_data:
        gdf_path_links = gpd.GeoDataFrame(path_links_data, geometry="geometry", crs="EPSG:4326")
    else:
        gdf_path_links = gpd.GeoDataFrame(columns=["from_node_id", "to_node_id", "geometry"], crs="EPSG:4326")

    if os.path.exists(gpkg_path):
        os.remove(gpkg_path)

    gdf_path_nodes.to_file(gpkg_path, layer="path_nodes", driver="GPKG")
    gdf_path_links.to_file(gpkg_path, layer="path_links", driver="GPKG", mode="a")

    print(f"  Shortest path exported to: {gpkg_path}")
    print(f"    - Layer 'path_nodes': {len(gdf_path_nodes)} nodes")
    print(f"    - Layer 'path_links': {len(gdf_path_links)} links")
    print(f"    - Links with route shapes: {n_with_shapes}")
    print(f"    - Links with straight lines: {len(gdf_path_links) - n_with_shapes}")
