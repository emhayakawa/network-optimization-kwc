"""
Build multimodal ION network combining:
1. GRT Bus routes from Raw_GTFS
2. ION LRT from GTFS(onlyLRT) + extension stops from ION_Stops.csv
3. Route geometry from ION_Routes.csv for proper coordinate fitting

Network structure:
- Nodes: (stop_id, route_id, mode) - one node per stop-route-mode combination
- Links:
  1. Route segments: consecutive stops on same route (travel time from GTFS)
  2. Transfer links: between routes at same stop/cluster
  3. Multimodal transfers: bus <-> LRT at nearby stops
"""
import os
import sys
import pandas as pd
import numpy as np

# Ensure URA root is in path for transit module
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

# Handle both module import and direct execution
if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from config import (
        LRT_GTFS_DIR,
        BUS_GTFS_DIR,
        ION_STOPS_CSV,
        ION_ROUTES_CSV,
        TRANSFER_TIME_BUS,
        TRANSFER_TIME_BUS_LRT,
        BUS_CLUSTER_RADIUS_M,
        MULTIMODAL_TRANSFER_RADIUS_M,
        FARE_CONSTANT,
        VALUE_OF_TIME,
        WAITING_TIME_BUS,
        WAITING_TIME_LRT,
        MODE_BUS,
        MODE_LRT,
        LRT_ROUTE_PREFIX,
        ION_STOPS_CRS,
    )
    from preprocess_lrt import merge_lrt_stops_with_gtfs, load_lrt_gtfs_stop_times, _parse_time, _haversine_m
else:
    from .config import (
        LRT_GTFS_DIR,
        BUS_GTFS_DIR,
        ION_STOPS_CSV,
        ION_ROUTES_CSV,
        TRANSFER_TIME_BUS,
        TRANSFER_TIME_BUS_LRT,
        BUS_CLUSTER_RADIUS_M,
        MULTIMODAL_TRANSFER_RADIUS_M,
        FARE_CONSTANT,
        VALUE_OF_TIME,
        WAITING_TIME_BUS,
        WAITING_TIME_LRT,
        MODE_BUS,
        MODE_LRT,
        LRT_ROUTE_PREFIX,
        ION_STOPS_CRS,
    )
    from .preprocess_lrt import merge_lrt_stops_with_gtfs, load_lrt_gtfs_stop_times, _parse_time, _haversine_m


def _parse_time_local(s):
    """Parse GTFS time 'H:MM:SS' or 'HH:MM:SS' to minutes since midnight."""
    if pd.isna(s) or s == "":
        return np.nan
    parts = str(s).strip().split(":")
    if len(parts) < 3:
        return np.nan
    h, m, sec = int(parts[0]), int(parts[1]), int(parts[2])
    return h * 60 + m + sec / 60.0


def _haversine_m_local(lat1, lon1, lat2, lon2):
    """Approximate distance in meters between two WGS84 points."""
    R = 6371000
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c


def load_bus_gtfs(gtfs_dir=None):
    """
    Load bus GTFS data from Raw_GTFS directory.
    
    Returns:
        dict with routes, stops, stop_times, trips DataFrames
    """
    path = gtfs_dir or BUS_GTFS_DIR
    data = {}
    
    for name in ["routes", "stops", "stop_times", "trips"]:
        fname = os.path.join(path, f"{name}.txt")
        if not os.path.exists(fname):
            raise FileNotFoundError(f"GTFS file not found: {fname}")
        data[name] = pd.read_csv(fname)
    
    # Ensure consistent types
    data["stops"]["stop_id"] = data["stops"]["stop_id"].astype(str)
    data["stop_times"]["stop_id"] = data["stop_times"]["stop_id"].astype(str)
    data["trips"]["route_id"] = data["trips"]["route_id"].astype(str)
    
    return data


def load_lrt_gtfs(gtfs_dir=None):
    """
    Load LRT GTFS data from GTFS(onlyLRT) directory.
    
    Returns:
        dict with routes, stops, stop_times, trips DataFrames
    """
    path = gtfs_dir or LRT_GTFS_DIR
    data = {}
    
    for name in ["routes", "stops", "stop_times", "trips"]:
        fname = os.path.join(path, f"{name}.txt")
        if not os.path.exists(fname):
            raise FileNotFoundError(f"GTFS file not found: {fname}")
        data[name] = pd.read_csv(fname)
    
    # Ensure consistent types and add LRT prefix to route_id
    data["stops"]["stop_id"] = data["stops"]["stop_id"].astype(str)
    data["stop_times"]["stop_id"] = data["stop_times"]["stop_id"].astype(str)
    data["trips"]["route_id"] = LRT_ROUTE_PREFIX + data["trips"]["route_id"].astype(str)
    data["routes"]["route_id"] = LRT_ROUTE_PREFIX + data["routes"]["route_id"].astype(str)
    
    return data


def load_ion_routes(csv_path=None):
    """
    Load ION route geometry from ION_Routes.csv.
    
    This contains route segments with stage information and status.
    Used to properly fit stops to routes.
    
    Returns:
        DataFrame with route segment information
    """
    path = csv_path or ION_ROUTES_CSV
    
    if not os.path.exists(path):
        print(f"  Warning: ION_Routes.csv not found at {path}")
        return pd.DataFrame()
    
    df = pd.read_csv(path)
    return df


def load_ion_stops(csv_path=None):
    """
    Load all ION stops from ION_Stops.csv.
    
    Includes both LRT stops (current and extension) and aBRT stops.
    
    Returns:
        DataFrame with stop information including coordinates
    """
    path = csv_path or ION_STOPS_CSV
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"ION stops file not found: {path}")
    
    df = pd.read_csv(path)
    
    # Rename columns for consistency
    df = df.rename(columns={
        "X": "x_utm",
        "Y": "y_utm",
        "StopName": "stop_name",
        "Municipality": "municipality",
        "StopLocation": "stop_location",
        "StopStatus": "stop_status",
        "StopDirection": "stop_direction",
        "Stage1": "stage1",
        "Stage2": "stage2",
    })
    
    # Convert UTM to WGS84
    df = _convert_utm_to_wgs84(df)
    
    # Categorize stops by mode
    df["mode"] = df.apply(_categorize_ion_stop, axis=1)
    
    return df


def _convert_utm_to_wgs84(df):
    """Convert UTM coordinates (x_utm, y_utm) to WGS84 (stop_lat, stop_lon)."""
    try:
        from pyproj import Transformer
        transformer = Transformer.from_crs(ION_STOPS_CRS, "EPSG:4326", always_xy=True)
        lons, lats = transformer.transform(df["x_utm"].values, df["y_utm"].values)
        df["stop_lon"] = lons
        df["stop_lat"] = lats
    except ImportError:
        # Approximate conversion for Waterloo region (UTM Zone 17N)
        df["stop_lon"] = (df["x_utm"] - 500000) / 111320 / np.cos(np.radians(43.5)) - 80.5
        df["stop_lat"] = df["y_utm"] / 111320
    return df


def _categorize_ion_stop(row):
    """Categorize an ION stop as LRT or aBRT based on stage info."""
    stage1 = str(row.get("stage1", "")).upper()
    stage2 = str(row.get("stage2", "")).upper()
    stop_name = str(row.get("stop_name", ""))
    
    # Check for LRT designation
    if "LRT" in stage1 or "LRT" in stage2 or "(LRT)" in stop_name:
        return MODE_LRT
    elif "ABRT" in stage1 or "ABRT" in stage2:
        return MODE_BUS  # aBRT treated as bus mode
    else:
        return MODE_BUS


def cluster_nearby_stops(stops_df, cluster_radius_m=None):
    """
    Cluster nearby stops within cluster_radius_m of each other.
    
    Returns:
        stops_df with cluster_id, cluster_lat, cluster_lon columns
    """
    if cluster_radius_m is None:
        cluster_radius_m = BUS_CLUSTER_RADIUS_M
    
    stops = stops_df.copy()
    stops["stop_id"] = stops["stop_id"].astype(str)
    
    stops["cluster_id"] = -1
    stops["cluster_lat"] = np.nan
    stops["cluster_lon"] = np.nan
    
    clusters = []
    next_cluster_id = 0
    
    for idx, row in stops.iterrows():
        lat, lon = float(row["stop_lat"]), float(row["stop_lon"])
        assigned = False
        
        for cluster in clusters:
            c_id, c_lat, c_lon, members = cluster
            dist = _haversine_m_local(lat, lon, c_lat, c_lon)
            if dist <= cluster_radius_m:
                members.append(idx)
                member_lats = stops.loc[members, "stop_lat"].astype(float).values
                member_lons = stops.loc[members, "stop_lon"].astype(float).values
                cluster[1] = np.mean(member_lats)
                cluster[2] = np.mean(member_lons)
                assigned = True
                break
        
        if not assigned:
            clusters.append([next_cluster_id, lat, lon, [idx]])
            next_cluster_id += 1
    
    for cluster in clusters:
        c_id, c_lat, c_lon, members = cluster
        for idx in members:
            stops.loc[idx, "cluster_id"] = c_id
            stops.loc[idx, "cluster_lat"] = c_lat
            stops.loc[idx, "cluster_lon"] = c_lon
    
    n_clustered = sum(1 for c in clusters if len(c[3]) > 1)
    n_stops_in_clusters = sum(len(c[3]) for c in clusters if len(c[3]) > 1)
    print(f"  Stop clustering: {n_clustered} clusters from {n_stops_in_clusters} stops (radius={cluster_radius_m}m)")
    
    return stops


def build_bus_network_component(bus_gtfs=None):
    """
    Build bus network nodes and links from Raw_GTFS.
    
    Returns:
        nodes_df, links_df for bus network
    """
    print("\n--- Loading Bus GTFS ---")
    data = load_bus_gtfs(bus_gtfs)
    stops = data["stops"]
    stop_times = data["stop_times"]
    trips = data["trips"]
    
    print(f"  Routes: {len(data['routes'])}, Stops: {len(stops)}, Trips: {len(trips)}")
    
    # Cluster nearby stops (same radius as bus_network for identical bus nodes/coordinates)
    stops = cluster_nearby_stops(stops, BUS_CLUSTER_RADIUS_M)
    
    # Join stop_times with trips to get route_id
    st = stop_times.merge(trips[["trip_id", "route_id"]], on="trip_id", how="left")
    st = st.sort_values(["trip_id", "stop_sequence"])
    
    # Unique (stop_id, route_id) pairs
    stop_route_pairs = st[["stop_id", "route_id"]].drop_duplicates().reset_index(drop=True)
    stop_route_pairs["node_id"] = np.arange(1, len(stop_route_pairs) + 1)
    stop_route_pairs["mode"] = MODE_BUS
    
    sr_to_node = {}
    for _, row in stop_route_pairs.iterrows():
        sr_to_node[(row["stop_id"], row["route_id"])] = int(row["node_id"])
    
    # Create nodes with coordinates
    stops_coords = stops[["stop_id", "stop_lat", "stop_lon", "cluster_id", "cluster_lat", "cluster_lon"]].drop_duplicates()
    nodes_df = stop_route_pairs.merge(stops_coords, on="stop_id", how="left")
    
    # Use cluster coordinates for node positions
    nodes_df["original_stop_lat"] = nodes_df["stop_lat"]
    nodes_df["original_stop_lon"] = nodes_df["stop_lon"]
    nodes_df["stop_lat"] = nodes_df["cluster_lat"]
    nodes_df["stop_lon"] = nodes_df["cluster_lon"]
    
    print(f"  Bus nodes: {len(nodes_df)}")
    
    # Build route segment links using ALL trips to capture all stop patterns
    st["arrival_min"] = st["arrival_time"].apply(_parse_time_local)
    st["departure_min"] = st["departure_time"].apply(_parse_time_local)
    
    # Collect all unique (from_stop, to_stop, route) segments with travel times
    segment_times = {}
    
    for trip_id, grp in st.groupby("trip_id"):
        grp = grp.sort_values("stop_sequence")
        stop_ids = grp["stop_id"].tolist()
        route_id = grp["route_id"].iloc[0]
        arr = grp["arrival_min"].values
        dep = grp["departure_min"].values
        
        for i in range(len(stop_ids) - 1):
            s1, s2 = stop_ids[i], stop_ids[i + 1]
            key = (s1, s2, route_id)
            travel_min = arr[i + 1] - dep[i]
            if not np.isnan(travel_min) and travel_min > 0:
                if key not in segment_times:
                    segment_times[key] = []
                segment_times[key].append(travel_min)
    
    # Create route links using average travel time for each segment
    route_links = []
    link_id = 1
    
    for (s1, s2, route_id), times in segment_times.items():
        n1 = sr_to_node.get((s1, route_id))
        n2 = sr_to_node.get((s2, route_id))
        if n1 is None or n2 is None:
            continue
        
        avg_travel_min = np.mean(times) if times else 2.0
        
        r1 = stops[stops["stop_id"] == s1].iloc[0]
        r2 = stops[stops["stop_id"] == s2].iloc[0]
        length_m = _haversine_m_local(
            float(r1["stop_lat"]), float(r1["stop_lon"]),
            float(r2["stop_lat"]), float(r2["stop_lon"])
        )
        
        route_links.append({
            "link_id": link_id,
            "from_node_id": n1,
            "to_node_id": n2,
            "link_type": "route",
            "route_id": route_id,
            "travel_time_min": round(avg_travel_min, 2),
            "length_m": round(length_m, 2),
            "mode": MODE_BUS,
        })
        link_id += 1
    
    # Create DataFrame (already unique by construction)
    links_df = pd.DataFrame(route_links)

    # Add reverse direction only when the route actually runs both ways (reverse segment exists in trips)
    rev_links = []
    node_to_stop = nodes_df.set_index("node_id")["stop_id"].to_dict()
    for _, row in links_df.iterrows():
        n1, n2 = row["from_node_id"], row["to_node_id"]
        s1, s2 = node_to_stop.get(n1), node_to_stop.get(n2)
        route_id = row["route_id"]
        if s1 is not None and s2 is not None and (str(s2), str(s1), route_id) in segment_times:
            rev_links.append({
                "from_node_id": n2,
                "to_node_id": n1,
                "link_type": "route",
                "route_id": route_id,
                "travel_time_min": row["travel_time_min"],
                "length_m": row["length_m"],
                "mode": MODE_BUS,
            })
    if rev_links:
        rev_df = pd.DataFrame(rev_links)
        rev_df["link_id"] = np.arange(len(links_df) + 1, len(links_df) + len(rev_df) + 1)
        links_df["link_id"] = np.arange(1, len(links_df) + 1)
        links_df = pd.concat([links_df, rev_df], ignore_index=True)
    else:
        links_df["link_id"] = np.arange(1, len(links_df) + 1)
    
    print(f"  Bus route links: {len(links_df)}")
    
    # Build transfer links within clusters
    cluster_to_nodes = nodes_df.groupby("cluster_id")["node_id"].apply(lambda x: x.astype(int).tolist()).to_dict()
    
    transfer_links = []
    next_link_id = links_df["link_id"].max() + 1
    
    node_routes = nodes_df.set_index("node_id")["route_id"].to_dict()
    
    for cluster_id, nids in cluster_to_nodes.items():
        if len(nids) <= 1:
            continue
        for i, n_from in enumerate(nids):
            for j, n_to in enumerate(nids):
                if i != j and node_routes.get(n_from) != node_routes.get(n_to):
                    transfer_links.append({
                        "link_id": next_link_id,
                        "from_node_id": n_from,
                        "to_node_id": n_to,
                        "link_type": "transfer",
                        "route_id": None,
                        "travel_time_min": TRANSFER_TIME_BUS,
                        "length_m": 0.0,
                        "mode": MODE_BUS,
                    })
                    next_link_id += 1
    
    transfer_links_df = pd.DataFrame(transfer_links)
    if len(transfer_links_df) > 0:
        links_df = pd.concat([links_df, transfer_links_df], ignore_index=True)
    
    print(f"  Bus transfer links: {len(transfer_links_df)}")
    print(f"  Total bus links: {len(links_df)}")
    
    # Clean up nodes columns
    nodes_df = nodes_df[["node_id", "stop_id", "route_id", "stop_lat", "stop_lon", "cluster_id", "mode"]]
    
    # Diagnostic: check for disconnected nodes
    all_node_ids = set(nodes_df["node_id"].astype(int))
    connected_from = set(links_df["from_node_id"].astype(int))
    connected_to = set(links_df["to_node_id"].astype(int))
    connected_nodes = connected_from | connected_to
    disconnected = all_node_ids - connected_nodes
    
    if disconnected:
        print(f"  WARNING: {len(disconnected)} bus nodes have no links (disconnected)")
        disconnected_df = nodes_df[nodes_df["node_id"].isin(disconnected)]
        routes_affected = disconnected_df["route_id"].unique()
        print(f"    Routes with disconnected nodes: {len(routes_affected)}")
    else:
        print(f"  All {len(all_node_ids)} bus nodes are connected")
    
    return nodes_df, links_df


def build_lrt_network_component(lrt_gtfs=None, ion_stops_csv=None, ion_routes_csv=None):
    """
    Build LRT network nodes and links from GTFS(onlyLRT) + ION extension stops.
    
    Uses ION_Routes.csv to properly fit stops to route geometry.
    
    Returns:
        nodes_df, links_df for LRT network
    """
    print("\n--- Loading LRT GTFS ---")
    data = load_lrt_gtfs(lrt_gtfs)
    stops = data["stops"]
    stop_times = data["stop_times"]
    trips = data["trips"]
    
    print(f"  LRT Routes: {len(data['routes'])}, Stops: {len(stops)}, Trips: {len(trips)}")
    
    # Load ION extension stops
    print("\n--- Loading ION Extension Stops ---")
    preprocess_result = merge_lrt_stops_with_gtfs(gtfs_dir=lrt_gtfs, ion_csv_path=ion_stops_csv)
    augmented_stops = preprocess_result["augmented_stops"]
    extension_links = preprocess_result["extension_links"]
    
    print(f"  Total LRT stops (GTFS + extension): {len(augmented_stops)}")
    
    # Load ION routes for geometry fitting
    print("\n--- Loading ION Routes ---")
    ion_routes = load_ion_routes(ion_routes_csv)
    if len(ion_routes) > 0:
        lrt_routes = ion_routes[
            (ion_routes["Stage1"] == "LRT") | (ion_routes["Stage2"] == "LRT")
        ]
        print(f"  ION route segments: {len(ion_routes)}")
        print(f"  LRT route segments: {len(lrt_routes)}")
    
    # Create nodes from augmented stops
    # Use single route_id for all LRT stops (ION_301)
    lrt_route_id = f"{LRT_ROUTE_PREFIX}301"
    
    nodes_df = augmented_stops[["stop_id", "stop_name", "stop_lat", "stop_lon"]].copy()
    nodes_df["node_id"] = np.arange(1, len(nodes_df) + 1)
    nodes_df["stop_id"] = nodes_df["stop_id"].astype(str)
    nodes_df["route_id"] = lrt_route_id
    nodes_df["mode"] = MODE_LRT
    nodes_df["cluster_id"] = -1  # LRT stops not clustered internally
    
    # Reorder columns
    nodes_df = nodes_df[["node_id", "stop_id", "route_id", "stop_lat", "stop_lon", "cluster_id", "mode"]]
    
    print(f"  LRT nodes: {len(nodes_df)}")
    
    # Build route links from GTFS stop_times
    stop_times["arrival_min"] = stop_times["arrival_time"].apply(_parse_time_local)
    stop_times["departure_min"] = stop_times["departure_time"].apply(_parse_time_local)
    stop_times = stop_times.sort_values(["trip_id", "stop_sequence"])
    
    # Create stop_id -> node_id lookup
    stop_to_node = nodes_df.set_index("stop_id")["node_id"].to_dict()
    
    # Build stop coordinates lookup
    stop_coords = nodes_df.set_index("stop_id")[["stop_lat", "stop_lon"]].to_dict("index")
    
    # Collect segments from GTFS
    segments = {}
    
    for trip_id, trip_df in stop_times.groupby("trip_id"):
        trip_df = trip_df.sort_values("stop_sequence")
        stop_ids = trip_df["stop_id"].tolist()
        arrivals = trip_df["arrival_min"].values
        departures = trip_df["departure_min"].values
        
        for i in range(len(stop_ids) - 1):
            from_stop = str(stop_ids[i])
            to_stop = str(stop_ids[i + 1])
            
            travel_time = arrivals[i + 1] - departures[i]
            if not np.isnan(travel_time) and travel_time > 0:
                key = (from_stop, to_stop)
                if key not in segments:
                    segments[key] = []
                segments[key].append(travel_time)
    
    # Create GTFS links with average travel time
    gtfs_links = []
    link_id = 1
    
    for (from_stop, to_stop), times in segments.items():
        avg_time = np.mean(times)
        
        n1 = stop_to_node.get(from_stop)
        n2 = stop_to_node.get(to_stop)
        
        if n1 is None or n2 is None:
            continue
        
        c1 = stop_coords.get(from_stop)
        c2 = stop_coords.get(to_stop)
        
        if c1 and c2:
            dist_m = _haversine_m_local(c1["stop_lat"], c1["stop_lon"],
                                        c2["stop_lat"], c2["stop_lon"])
        else:
            dist_m = 0
        
        gtfs_links.append({
            "link_id": link_id,
            "from_node_id": n1,
            "to_node_id": n2,
            "link_type": "route",
            "route_id": lrt_route_id,
            "travel_time_min": round(avg_time, 2),
            "length_m": round(dist_m, 2),
            "mode": MODE_LRT,
        })
        link_id += 1
    
    gtfs_links_df = pd.DataFrame(gtfs_links)
    print(f"  GTFS LRT links: {len(gtfs_links_df)}")
    
    # Add extension links (from preprocess_lrt)
    if len(extension_links) > 0:
        extension_links["mode"] = MODE_LRT
        extension_links["route_id"] = lrt_route_id
        extension_links["link_type"] = "route"
        
        # Map stop_ids to node_ids
        extension_links["from_node_id"] = extension_links["from_stop_id"].astype(str).map(stop_to_node)
        extension_links["to_node_id"] = extension_links["to_stop_id"].astype(str).map(stop_to_node)
        
        # Remove rows with missing node mappings
        extension_links = extension_links.dropna(subset=["from_node_id", "to_node_id"])
        extension_links["from_node_id"] = extension_links["from_node_id"].astype(int)
        extension_links["to_node_id"] = extension_links["to_node_id"].astype(int)
        extension_links["link_id"] = np.arange(len(gtfs_links_df) + 1, len(gtfs_links_df) + len(extension_links) + 1)
        
        # Select columns
        ext_cols = ["link_id", "from_node_id", "to_node_id", "link_type", "route_id", 
                    "travel_time_min", "length_m", "mode"]
        extension_links = extension_links[ext_cols]
        
        print(f"  Extension links: {len(extension_links)}")
    else:
        extension_links = pd.DataFrame()
    
    # Combine GTFS + extension links
    # GTFS segments and extension links already include both directions when they exist in trip data
    links_parts = [gtfs_links_df]
    if len(extension_links) > 0:
        links_parts.append(extension_links)
    
    links_df = pd.concat(links_parts, ignore_index=True)
    links_df["link_id"] = np.arange(1, len(links_df) + 1)
    
    print(f"  Total LRT links: {len(links_df)}")
    
    # Diagnostic: check for disconnected nodes
    all_node_ids = set(nodes_df["node_id"].astype(int))
    connected_from = set(links_df["from_node_id"].astype(int))
    connected_to = set(links_df["to_node_id"].astype(int))
    connected_nodes = connected_from | connected_to
    disconnected = all_node_ids - connected_nodes
    
    if disconnected:
        print(f"  WARNING: {len(disconnected)} LRT nodes have no links (disconnected)")
    else:
        print(f"  All {len(all_node_ids)} LRT nodes are connected")
    
    return nodes_df, links_df


def merge_networks(bus_nodes, bus_links, lrt_nodes, lrt_links):
    """
    Merge bus and LRT networks into a combined multimodal network.
    
    Adds multimodal transfer links between bus and LRT at nearby stops.
    
    Returns:
        combined_nodes, combined_links
    """
    print("\n--- Merging Networks ---")
    
    # Offset LRT node IDs to avoid conflicts with bus
    max_bus_node_id = bus_nodes["node_id"].max()
    max_bus_link_id = bus_links["link_id"].max()
    
    lrt_nodes = lrt_nodes.copy()
    lrt_links = lrt_links.copy()
    
    # Create node ID mapping for LRT
    lrt_node_offset = max_bus_node_id
    old_to_new_lrt = {old: old + lrt_node_offset for old in lrt_nodes["node_id"]}
    
    lrt_nodes["node_id"] = lrt_nodes["node_id"] + lrt_node_offset
    lrt_links["from_node_id"] = lrt_links["from_node_id"] + lrt_node_offset
    lrt_links["to_node_id"] = lrt_links["to_node_id"] + lrt_node_offset
    lrt_links["link_id"] = lrt_links["link_id"] + max_bus_link_id
    
    # Combine nodes
    combined_nodes = pd.concat([bus_nodes, lrt_nodes], ignore_index=True)
    print(f"  Combined nodes: {len(combined_nodes)} (bus: {len(bus_nodes)}, LRT: {len(lrt_nodes)})")
    
    # Combine links
    combined_links = pd.concat([bus_links, lrt_links], ignore_index=True)
    
    # Build multimodal transfer links (bus <-> LRT at nearby stops)
    print("\n--- Building Multimodal Transfers ---")
    
    bus_stop_nodes = bus_nodes[["node_id", "stop_id", "stop_lat", "stop_lon", "mode"]].copy()
    lrt_stop_nodes = lrt_nodes[["node_id", "stop_id", "stop_lat", "stop_lon", "mode"]].copy()
    
    multimodal_links = []
    next_link_id = combined_links["link_id"].max() + 1
    
    # For each LRT stop, find nearby bus stops
    for _, lrt_row in lrt_stop_nodes.iterrows():
        lrt_lat = float(lrt_row["stop_lat"])
        lrt_lon = float(lrt_row["stop_lon"])
        lrt_node = int(lrt_row["node_id"])
        
        for _, bus_row in bus_stop_nodes.iterrows():
            bus_lat = float(bus_row["stop_lat"])
            bus_lon = float(bus_row["stop_lon"])
            bus_node = int(bus_row["node_id"])
            
            dist = _haversine_m_local(lrt_lat, lrt_lon, bus_lat, bus_lon)
            
            # Create transfer if within multimodal transfer radius
            if dist <= MULTIMODAL_TRANSFER_RADIUS_M:
                # Bus -> LRT
                multimodal_links.append({
                    "link_id": next_link_id,
                    "from_node_id": bus_node,
                    "to_node_id": lrt_node,
                    "link_type": "multimodal_transfer",
                    "route_id": None,
                    "travel_time_min": TRANSFER_TIME_BUS_LRT,
                    "length_m": round(dist, 2),
                    "mode": "transfer",
                })
                next_link_id += 1
                
                # LRT -> Bus
                multimodal_links.append({
                    "link_id": next_link_id,
                    "from_node_id": lrt_node,
                    "to_node_id": bus_node,
                    "link_type": "multimodal_transfer",
                    "route_id": None,
                    "travel_time_min": TRANSFER_TIME_BUS_LRT,
                    "length_m": round(dist, 2),
                    "mode": "transfer",
                })
                next_link_id += 1
    
    if multimodal_links:
        multimodal_df = pd.DataFrame(multimodal_links)
        combined_links = pd.concat([combined_links, multimodal_df], ignore_index=True)
        print(f"  Multimodal transfer links: {len(multimodal_links)}")
    else:
        print("  No multimodal transfers created (no nearby stops)")
    
    # Renumber link IDs
    combined_links["link_id"] = np.arange(1, len(combined_links) + 1)
    
    print(f"  Combined links: {len(combined_links)}")
    
    # Summary by link type
    print("\n  Link type summary:")
    for link_type, count in combined_links["link_type"].value_counts().items():
        print(f"    - {link_type}: {count}")
    
    # Final diagnostic: check for disconnected nodes in combined network
    all_node_ids = set(combined_nodes["node_id"].astype(int))
    connected_from = set(combined_links["from_node_id"].astype(int))
    connected_to = set(combined_links["to_node_id"].astype(int))
    connected_nodes = connected_from | connected_to
    disconnected = all_node_ids - connected_nodes
    
    if disconnected:
        print(f"\n  WARNING: {len(disconnected)} nodes have no links in combined network")
        disconnected_df = combined_nodes[combined_nodes["node_id"].isin(disconnected)]
        for mode, count in disconnected_df["mode"].value_counts().items():
            print(f"    - {mode}: {count} disconnected")
    else:
        print(f"\n  All {len(all_node_ids)} nodes are connected in combined network")
    
    return combined_nodes, combined_links


def build_nodes_and_links(
    bus_gtfs=None,
    lrt_gtfs=None,
    ion_stops_csv=None,
    ion_routes_csv=None,
    taz_path=None,
    assign_zone_id=True,
):
    """
    Build complete multimodal network (bus + LRT) nodes and links.

    When assign_zone_id is True and the TAZ shapefile exists, every node gets zone_id from
    (stop_lat, stop_lon) via point-in-polygon (same as bus_network).

    Returns:
        nodes_df, links_df
    """
    print("\n=== Building Bus Network Component ===")
    bus_nodes, bus_links = build_bus_network_component(bus_gtfs)
    
    print("\n=== Building LRT Network Component ===")
    lrt_nodes, lrt_links = build_lrt_network_component(lrt_gtfs, ion_stops_csv, ion_routes_csv)
    
    print("\n=== Merging into Multimodal Network ===")
    nodes_df, links_df = merge_networks(bus_nodes, bus_links, lrt_nodes, lrt_links)

    if assign_zone_id:
        from bus_network.config import DEFAULT_TAZ_SHAPEFILE
        from bus_network.zones import assign_zone_id_by_location, load_taz_zones

        taz_file = taz_path or DEFAULT_TAZ_SHAPEFILE
        if os.path.isfile(taz_file):
            zones_gdf = load_taz_zones(taz_file)
            nodes_df = assign_zone_id_by_location(
                nodes_df, zones_gdf, lat_col="stop_lat", lon_col="stop_lon"
            )
        else:
            print(
                f"  TAZ file not found ({taz_file}); skipping zone_id on ION network nodes"
            )

    return nodes_df, links_df


def save_network_data(nodes_df, links_df, link_shapes, out_dir, srid=26917, verbose=True):
    """
    Save ION network to node.csv, link.csv, geometry.csv (canonical output for ArcGIS and AequilibraE).
    
    Geometry uses link_shapes (GTFS route shapes) for route links, straight lines for others.
    """
    import geopandas as gpd
    from shapely.geometry import LineString
    from shapely.ops import transform
    import pyproj

    os.makedirs(out_dir, exist_ok=True)
    
    gdf_nodes = gpd.GeoDataFrame(
        nodes_df,
        geometry=gpd.points_from_xy(nodes_df["stop_lon"], nodes_df["stop_lat"]),
        crs="EPSG:4326",
    )
    gdf_nodes = gdf_nodes.to_crs(f"EPSG:{srid}")
    gdf_nodes["x_coord"] = gdf_nodes.geometry.x
    gdf_nodes["y_coord"] = gdf_nodes.geometry.y
    if "cluster_id" in gdf_nodes.columns:
        delta_m = 0.5
        for cluster_id, idxs in gdf_nodes.groupby("cluster_id").groups.items():
            if len(idxs) <= 1:
                continue
            for i, idx in enumerate(idxs):
                gdf_nodes.loc[idx, "x_coord"] = gdf_nodes.loc[idx, "x_coord"] + (i % 10) * delta_m
                gdf_nodes.loc[idx, "y_coord"] = gdf_nodes.loc[idx, "y_coord"] + (i // 10) * delta_m
    
    node_file = os.path.join(out_dir, "node.csv")
    node_cols = ["node_id", "x_coord", "y_coord"]
    extra = [
        c
        for c in [
            "stop_id",
            "route_id",
            "stop_lat",
            "stop_lon",
            "cluster_id",
            "zone_id",
            "mode",
        ]
        if c in gdf_nodes.columns
    ]
    gdf_nodes[node_cols + extra].to_csv(node_file, index=False)
    
    # Links: match links.csv schema (link_id, from_node_id, to_node_id, link_type, route_id, travel_time_min, length_m, mode)
    link_cols = ["link_id", "from_node_id", "to_node_id", "link_type", "route_id", "travel_time_min", "length_m", "mode"]
    link_cols = [c for c in link_cols if c in links_df.columns]
    links_export = links_df[link_cols].copy()
    link_file = os.path.join(out_dir, "link.csv")
    links_export.to_csv(link_file, index=False)
    
    # For transfer/multimodal links: use display coords so zero-length links get visible geometry
    node_xy = gdf_nodes.set_index("node_id")[["x_coord", "y_coord"]].to_dict("index")
    node_coord = nodes_df.set_index("node_id")[["stop_lat", "stop_lon"]].to_dict("index")
    transformer = pyproj.Transformer.from_crs("EPSG:4326", f"EPSG:{srid}", always_xy=True)
    TRANSFER_LINK_MIN_M = 2.0  # min visible length for zero-length links
    geoms = []
    n_with_shapes = 0
    for _, row in links_df.iterrows():
        f, t = int(row["from_node_id"]), int(row["to_node_id"])
        shape_coords = link_shapes.get((f, t)) if link_shapes else None
        if shape_coords and len(shape_coords) >= 2:
            line = LineString(shape_coords)
            n_with_shapes += 1
            line_proj = transform(transformer.transform, line)
        else:
            xy1 = node_xy.get(f)
            xy2 = node_xy.get(t)
            if xy1 and xy2:
                x1, y1 = float(xy1["x_coord"]), float(xy1["y_coord"])
                x2, y2 = float(xy2["x_coord"]), float(xy2["y_coord"])
                line_proj = LineString([(x1, y1), (x2, y2)])
                # Ensure transfer links are visible in ArcGIS (avoid zero-length)
                if row.get("link_type") in ("transfer", "multimodal_transfer"):
                    dx, dy = x2 - x1, y2 - y1
                    d = (dx**2 + dy**2) ** 0.5
                    if d < TRANSFER_LINK_MIN_M:
                        if d < 1e-6:
                            x2, y2 = x1 + TRANSFER_LINK_MIN_M, y1
                        else:
                            x2 = x1 + dx * (TRANSFER_LINK_MIN_M / d)
                            y2 = y1 + dy * (TRANSFER_LINK_MIN_M / d)
                        line_proj = LineString([(x1, y1), (x2, y2)])
            else:
                c1, c2 = node_coord.get(f), node_coord.get(t)
                if c1 and c2:
                    line = LineString([(float(c1["stop_lon"]), float(c1["stop_lat"])), (float(c2["stop_lon"]), float(c2["stop_lat"]))])
                    line_proj = transform(transformer.transform, line)
                else:
                    continue
        geoms.append({"geometry_id": row["link_id"], "geometry": line_proj.wkt})
    geom_df = pd.DataFrame(geoms)
    geometry_file = os.path.join(out_dir, "geometry.csv")
    geom_df.to_csv(geometry_file, index=False)
    
    if verbose:
        print(f"  Saved to {out_dir}:")
        print(f"    - node.csv: {len(gdf_nodes)} nodes")
        print(f"    - link.csv: {len(links_df)} links")
        print(f"    - geometry.csv: {len(geom_df)} geometries ({n_with_shapes} with route shapes)")
    return node_file, link_file, geometry_file


def export_to_arcgis_from_data(data_dir, gpkg_path, verbose=True):
    """
    Create ArcGIS GeoPackage from saved node.csv, link.csv, geometry.csv.
    Columns in node.csv (e.g. zone_id) are carried into the nodes layer.
    """
    from shapely import wkt

    node_file = os.path.join(data_dir, "node.csv")
    link_file = os.path.join(data_dir, "link.csv")
    geometry_file = os.path.join(data_dir, "geometry.csv")
    for f in [node_file, link_file, geometry_file]:
        if not os.path.exists(f):
            raise FileNotFoundError(f"Missing {f} - run build_network first")
    
    nodes_df = pd.read_csv(node_file)
    links_df = pd.read_csv(link_file)
    geom_df = pd.read_csv(geometry_file)
    
    import geopandas as gpd
    gdf_nodes = gpd.GeoDataFrame(
        nodes_df,
        geometry=gpd.points_from_xy(nodes_df["x_coord"], nodes_df["y_coord"]),
        crs="EPSG:26917",
    )
    gdf_nodes = gdf_nodes.to_crs("EPSG:4326")
    
    geom_lookup = geom_df.set_index("geometry_id")["geometry"].to_dict()
    link_geoms = []
    for _, row in links_df.iterrows():
        wkt_str = geom_lookup.get(row["link_id"])
        link_geoms.append(wkt.loads(wkt_str) if wkt_str else None)
    links_df = links_df.copy()
    links_df["geometry"] = link_geoms
    gdf_links = gpd.GeoDataFrame(
        links_df.dropna(subset=["geometry"]),
        geometry="geometry",
        crs="EPSG:26917",
    )
    gdf_links = gdf_links.to_crs("EPSG:4326")
    
    os.makedirs(os.path.dirname(gpkg_path) or ".", exist_ok=True)
    if os.path.exists(gpkg_path):
        os.remove(gpkg_path)
    gdf_nodes.to_file(gpkg_path, layer="nodes", driver="GPKG")
    gdf_links.to_file(gpkg_path, layer="links", driver="GPKG", mode="a")
    
    if verbose:
        print(f"  ArcGIS export: {gpkg_path}")
        print(f"    - Nodes: {len(gdf_nodes)}, Links: {len(gdf_links)}")


def load_gtfs_shapes(gtfs_dir):
    """
    Load shapes.txt from GTFS directory.
    
    Returns:
        dict: shape_id -> list of (lon, lat) coordinates in sequence order
    """
    shapes_file = os.path.join(gtfs_dir, "shapes.txt")
    if not os.path.exists(shapes_file):
        return {}
    
    shapes_df = pd.read_csv(shapes_file)
    shapes_df = shapes_df.sort_values(["shape_id", "shape_pt_sequence"])
    
    shape_coords = {}
    for shape_id, grp in shapes_df.groupby("shape_id"):
        coords = list(zip(grp["shape_pt_lon"].values, grp["shape_pt_lat"].values))
        shape_coords[str(shape_id)] = coords
    
    return shape_coords


def build_link_shape_lookup(gtfs_dir, nodes_df, links_df, route_prefix=""):
    """
    Build a lookup from (from_node_id, to_node_id) to shape geometry.
    
    For each route link, finds the segment of the shape between the two stops.
    Uses trip-specific shapes when routes have multiple shapes (e.g., directions).
    
    Parameters:
        gtfs_dir: Path to GTFS directory
        nodes_df: Network nodes
        links_df: Network links
        route_prefix: Prefix added to route_ids (e.g., "ION_" for LRT)
    
    Returns:
        dict: (from_node_id, to_node_id) -> list of (lon, lat) coordinates
    """
    # Load shapes
    shape_coords = load_gtfs_shapes(gtfs_dir)
    if not shape_coords:
        return {}
    
    trips_file = os.path.join(gtfs_dir, "trips.txt")
    if not os.path.exists(trips_file):
        return {}
    
    trips_df = pd.read_csv(trips_file)
    trips_df["route_id"] = trips_df["route_id"].astype(str)
    trips_df["shape_id"] = trips_df["shape_id"].astype(str)
    
    # Build (from_stop, to_stop, route_id) -> shape_id from trips that serve both stops in order
    segment_to_shape = {}
    stop_times_file = os.path.join(gtfs_dir, "stop_times.txt")
    if os.path.exists(stop_times_file):
        stop_times = pd.read_csv(stop_times_file)
        stop_times["stop_id"] = stop_times["stop_id"].astype(str)
        st_trips = stop_times.merge(trips_df[["trip_id", "route_id", "shape_id"]], on="trip_id", how="left")
        for (route_id, trip_id), grp in st_trips.groupby(["route_id", "trip_id"]):
            grp = grp.sort_values("stop_sequence")
            stops = grp["stop_id"].tolist()
            shape_id = grp["shape_id"].iloc[0]
            for i in range(len(stops) - 1):
                key = (stops[i], stops[i + 1], str(route_id))
                if key not in segment_to_shape:
                    segment_to_shape[key] = shape_id
    
    # Fallback: one shape per route
    route_to_shape = {}
    for route_id, shape_id in trips_df.groupby("route_id")["shape_id"].first().items():
        route_to_shape[route_prefix + str(route_id)] = shape_id
    
    node_to_stop = nodes_df.set_index("node_id")["stop_id"].astype(str).to_dict()
    node_coords = nodes_df.set_index("node_id")[["stop_lat", "stop_lon"]].to_dict("index")
    
    # Build link shape lookup
    link_shapes = {}
    
    route_links = links_df[links_df["link_type"] == "route"].copy()
    
    for _, link in route_links.iterrows():
        from_node = int(link["from_node_id"])
        to_node = int(link["to_node_id"])
        route_id = link.get("route_id")
        
        if pd.isna(route_id):
            continue
        
        route_id = str(route_id)
        from_stop = node_to_stop.get(from_node)
        to_stop = node_to_stop.get(to_node)
        # Base route for GTFS lookup (without prefix)
        base_route = route_id[len(route_prefix):] if route_prefix and route_id.startswith(route_prefix) else route_id
        shape_id = None
        if from_stop and to_stop:
            shape_id = segment_to_shape.get((from_stop, to_stop, base_route))
        if not shape_id:
            shape_id = route_to_shape.get(route_id)
        
        if not shape_id or shape_id not in shape_coords:
            continue
        
        coords = shape_coords[shape_id]
        if len(coords) < 2:
            continue
        
        # Get stop coordinates
        c1 = node_coords.get(from_node)
        c2 = node_coords.get(to_node)
        
        if not c1 or not c2:
            continue
        
        from_pt = (float(c1["stop_lon"]), float(c1["stop_lat"]))
        to_pt = (float(c2["stop_lon"]), float(c2["stop_lat"]))
        
        # Find closest points on shape to each stop
        from_idx = _find_closest_point_idx(coords, from_pt)
        to_idx = _find_closest_point_idx(coords, to_pt)
        
        # Extract segment (handle both directions)
        if from_idx <= to_idx:
            segment = list(coords[from_idx:to_idx + 1])
        else:
            segment = list(coords[to_idx:from_idx + 1][::-1])
        
        # Ensure we have at least 2 points
        if len(segment) < 2:
            segment = [from_pt, to_pt]
        else:
            # Ensure link passes through nodes (stops may be offset from road centerline)
            segment[0] = from_pt
            segment[-1] = to_pt
        
        link_shapes[(from_node, to_node)] = segment
    
    # For reverse route links: use reversed forward segment when forward has proper route geometry.
    for _, link in route_links.iterrows():
        from_node = int(link["from_node_id"])
        to_node = int(link["to_node_id"])
        rev_segment = link_shapes.get((to_node, from_node))
        if rev_segment and len(rev_segment) >= 3:
            existing = link_shapes.get((from_node, to_node))
            if not existing or len(existing) < 3:
                link_shapes[(from_node, to_node)] = list(reversed(rev_segment))
    
    return link_shapes


def _find_closest_point_idx(coords, target):
    """Find index of closest point in coords to target (lon, lat)."""
    min_dist = float("inf")
    min_idx = 0
    
    for i, (lon, lat) in enumerate(coords):
        dist = (lon - target[0])**2 + (lat - target[1])**2
        if dist < min_dist:
            min_dist = dist
            min_idx = i
    
    return min_idx


# Extension stop IDs (Fairway + Cambridge extension)
_EXTENSION_STOP_IDS = {"6108", "6016", "10001", "10002", "10003", "10004", "10005", "10006", "10007"}

# Max distance (m) from stop to path for "passes through" verification
_VERIFY_STOP_TOLERANCE_M = 150


def _haversine_m(lat1, lon1, lat2, lon2):
    """Distance in meters between two WGS84 points."""
    R = 6371000
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c


def verify_extension_geometry_passes_through_stops(extension_coords, stops_df, tolerance_m=None):
    """
    Verify that the extension path (from GeoJSON) passes through all extension stops.
    Returns dict of stop_id -> (distance_m, passed).
    """
    if tolerance_m is None:
        tolerance_m = _VERIFY_STOP_TOLERANCE_M
    if len(extension_coords) < 2:
        return {}

    ext_stops = stops_df[stops_df["stop_id"].astype(str).isin(_EXTENSION_STOP_IDS)]
    results = {}
    for _, row in ext_stops.iterrows():
        sid = str(row["stop_id"])
        lat, lon = float(row["stop_lat"]), float(row["stop_lon"])
        min_d = float("inf")
        for pt in extension_coords:
            d = _haversine_m(lat, lon, pt[1], pt[0])
            min_d = min(min_d, d)
        results[sid] = (min_d, min_d <= tolerance_m)
    return results


def _load_extension_geometry_from_geojson(geojson_path):
    """
    Load LRT route geometry from ION_Routes.geojson for the Cambridge extension corridor.
    Returns a single ordered list of (lon, lat) forming the path from Fairway to Cambridge Terminus.
    """
    import json

    if not geojson_path or not os.path.exists(geojson_path):
        return []

    with open(geojson_path) as f:
        data = json.load(f)

    # Extension corridor: Fairway (43.42, -80.44) to Cambridge (43.36, -80.31)
    ext_bbox = (43.35, 43.43, -80.46, -80.30)  # min_lat, max_lat, min_lon, max_lon

    def in_extension(coords):
        lons = [c[0] for c in coords]
        lats = [c[1] for c in coords]
        return (min(lats) < ext_bbox[1] and max(lats) > ext_bbox[0] and
                min(lons) > ext_bbox[2] and max(lons) < ext_bbox[3])

    # Only Stage2=LRT where Stage1 is not LRT: extension corridor (Fairway to Cambridge).
    # Stage1=LRT (Conestoga to Fairway) is already in GTFS(withLRT).
    def is_stage2_extension(props):
        s1, s2 = props.get("Stage1"), props.get("Stage2")
        return s2 == "LRT" and s1 != "LRT"

    segments = []
    for feat in data.get("features", []):
        if not is_stage2_extension(feat.get("properties", {})):
            continue
        geom = feat.get("geometry", {})
        if geom.get("type") == "LineString":
            coords = list(geom.get("coordinates", []))
        elif geom.get("type") == "MultiLineString":
            coords = []
            for part in geom.get("coordinates", []):
                coords.extend(part)
        else:
            continue
        if len(coords) >= 2 and in_extension(coords):
            segments.append(coords)

    if not segments:
        return []

    # Stitch segments into one path (north to south: Fairway 43.42 -> Cambridge 43.36)
    def dist_sq(a, b):
        return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2

    path = []
    used = [False] * len(segments)
    # Start with northernmost segment (highest avg lat)
    best_idx = max(range(len(segments)), key=lambda i: sum(c[1] for c in segments[i]) / len(segments[i]))
    path = list(segments[best_idx])
    used[best_idx] = True

    while True:
        end_pt = path[-1]
        best_next = None
        best_dist = float("inf")
        for i, seg in enumerate(segments):
            if used[i]:
                continue
            d1, d2 = dist_sq(seg[0], end_pt), dist_sq(seg[-1], end_pt)
            if d1 < best_dist:
                best_dist, best_next = d1, (i, list(reversed(seg)))
            if d2 < best_dist:
                best_dist, best_next = d2, (i, list(seg))
        if best_next is None or best_dist > 0.01:  # 0.01 deg ~ 1km gap tolerance
            break
        idx, coords = best_next
        used[idx] = True
        path.extend(coords[1:])

    return path


def build_extension_link_shapes(nodes_df, links_df, geojson_path):
    """
    Build link shapes for ION extension links using geometry from ION_Routes.geojson.
    Extension links connect Fairway (6108/6016) to Sportsworld (10001) and sequential extension stops.
    """
    extension_coords = _load_extension_geometry_from_geojson(geojson_path)
    if len(extension_coords) < 2:
        return {}

    node_to_stop = nodes_df.set_index("node_id")["stop_id"].astype(str).to_dict()
    node_coords = nodes_df.set_index("node_id")[["stop_lat", "stop_lon"]].to_dict("index")

    link_shapes = {}
    route_links = links_df[links_df["link_type"] == "route"]

    for _, link in route_links.iterrows():
        from_node = int(link["from_node_id"])
        to_node = int(link["to_node_id"])
        from_stop = node_to_stop.get(from_node)
        to_stop = node_to_stop.get(to_node)
        if not from_stop or not to_stop:
            continue
        if from_stop not in _EXTENSION_STOP_IDS or to_stop not in _EXTENSION_STOP_IDS:
            continue

        c1 = node_coords.get(from_node)
        c2 = node_coords.get(to_node)
        if not c1 or not c2:
            continue

        from_pt = (float(c1["stop_lon"]), float(c1["stop_lat"]))
        to_pt = (float(c2["stop_lon"]), float(c2["stop_lat"]))

        from_idx = _find_closest_point_idx(extension_coords, from_pt)
        to_idx = _find_closest_point_idx(extension_coords, to_pt)

        if from_idx <= to_idx:
            segment = list(extension_coords[from_idx : to_idx + 1])
        else:
            segment = list(extension_coords[to_idx : from_idx + 1][::-1])

        if len(segment) < 2:
            segment = [from_pt, to_pt]
        else:
            # Ensure link passes through stops when path does not (GeoJSON may not reach Fairway/Delta)
            d_from = _haversine_m(c1["stop_lat"], c1["stop_lon"], segment[0][1], segment[0][0])
            d_to = _haversine_m(c2["stop_lat"], c2["stop_lon"], segment[-1][1], segment[-1][0])
            if d_from > _VERIFY_STOP_TOLERANCE_M:
                segment = [from_pt] + segment
            if d_to > _VERIFY_STOP_TOLERANCE_M:
                segment = segment + [to_pt]

        link_shapes[(from_node, to_node)] = segment

    # Reverse shapes for opposite direction
    for (fn, tn), seg in list(link_shapes.items()):
        if (tn, fn) not in link_shapes and len(seg) >= 3:
            link_shapes[(tn, fn)] = list(reversed(seg))

    return link_shapes


def export_to_arcgis(nodes_df, links_df, out_path, bus_gtfs=None, lrt_gtfs=None, ion_routes_geojson=None, format="gpkg"):
    """
    Export network to GeoPackage for ArcGIS/QGIS visualization.
    
    Uses GTFS shapes.txt for route link geometry and ION_Routes.geojson for extension.
    """
    import geopandas as gpd
    from shapely.geometry import LineString, Point

    # Extension links use straight lines when ion_routes_geojson is not provided
    if len(nodes_df) == 0:
        print("No nodes to export")
        return

    nodes_df = nodes_df.copy()
    links_df = links_df.copy()

    # Build shape lookup for route links
    link_shapes = {}
    if bus_gtfs:
        print("  Loading bus route shapes...")
        bus_shapes = build_link_shape_lookup(bus_gtfs, nodes_df, links_df, route_prefix="")
        link_shapes.update(bus_shapes)
        print(f"    Found shapes for {len(bus_shapes)} bus links")

    if lrt_gtfs:
        print("  Loading LRT route shapes...")
        lrt_shapes = build_link_shape_lookup(lrt_gtfs, nodes_df, links_df, route_prefix=LRT_ROUTE_PREFIX)
        link_shapes.update(lrt_shapes)
        print(f"    Found shapes for {len(lrt_shapes)} LRT links")

    ext_shapes = build_extension_link_shapes(nodes_df, links_df, ion_routes_geojson) if ion_routes_geojson else {}
    if ext_shapes:
        link_shapes.update(ext_shapes)
        print(f"    Found shapes for {len(ext_shapes)} extension links")
    
    node_coord = nodes_df.set_index("node_id")[["stop_lat", "stop_lon"]].to_dict("index")
    
    gdf_nodes = gpd.GeoDataFrame(
        nodes_df,
        geometry=[Point(float(row["stop_lon"]), float(row["stop_lat"])) for _, row in nodes_df.iterrows()],
        crs="EPSG:4326",
    )
    
    geoms = []
    for _, row in links_df.iterrows():
        f, t = int(row["from_node_id"]), int(row["to_node_id"])
        
        # Try to get shape geometry first
        shape_coords = link_shapes.get((f, t))
        
        if shape_coords and len(shape_coords) >= 2:
            # Use actual shape geometry
            geoms.append(LineString(shape_coords))
        else:
            # Fall back to straight line
            c1 = node_coord.get(f)
            c2 = node_coord.get(t)
            if c1 and c2:
                geoms.append(LineString([
                    (float(c1["stop_lon"]), float(c1["stop_lat"])),
                    (float(c2["stop_lon"]), float(c2["stop_lat"]))
                ]))
            else:
                geoms.append(None)
    
    links_df["geometry"] = geoms
    gdf_links = gpd.GeoDataFrame(links_df.dropna(subset=["geometry"]), geometry="geometry", crs="EPSG:4326")
    
    out_path = os.path.abspath(out_path)
    if not out_path.endswith(".gpkg"):
        out_path = out_path + ".gpkg"
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    
    if os.path.exists(out_path):
        os.remove(out_path)
    
    gdf_nodes.to_file(out_path, layer="nodes", driver="GPKG")
    gdf_links.to_file(out_path, layer="links", driver="GPKG", mode="a")
    
    print(f"  Exported to: {out_path}")
    print(f"    - Layer 'nodes': {len(gdf_nodes)} nodes")
    print(f"    - Layer 'links': {len(gdf_links)} links")
    
    # Count links with shape geometry vs straight lines
    n_with_shapes = sum(1 for k in link_shapes if link_shapes[k])
    print(f"    - Links with route shapes: {n_with_shapes}")
    print(f"    - Links with straight lines: {len(gdf_links) - n_with_shapes}")


def build_network(
    bus_gtfs=None,
    lrt_gtfs=None,
    ion_stops_csv=None,
    ion_routes_csv=None,
    taz_path=None,
    assign_zone_id=True,
    export_arcgis=True,
    export_aequilibrae=False,
    verbose=True,
):
    """
    Main workflow: build the multimodal ION network (bus + LRT).
    
    Outputs node.csv, link.csv, geometry.csv to data/ (canonical format).
    ArcGIS gpkg is created FROM these saved files when export_arcgis=True.
    
    Parameters:
        bus_gtfs: Path to bus GTFS directory (default: Raw_GTFS)
        lrt_gtfs: Path to LRT GTFS directory (default: GTFS(onlyLRT))
        ion_stops_csv: Path to ION_Stops.csv
        ion_routes_csv: Path to ION_Routes.csv
        taz_path: TAZ polygons for zone_id on nodes (default: same as bus_network)
        assign_zone_id: attach zone_id to each node when the TAZ file exists
        export_arcgis: Create GeoPackage from saved data for visualization
        export_aequilibrae: Create AequilibraE project (for traffic assignment merge)
        verbose: Print progress
    
    Returns:
        nodes_df, links_df
    """
    if bus_gtfs is None:
        bus_gtfs = BUS_GTFS_DIR
    if lrt_gtfs is None:
        lrt_gtfs = LRT_GTFS_DIR
    
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    arcgis_dir = os.path.join(os.path.dirname(__file__), "arcgis_export")
    gpkg_path = os.path.join(arcgis_dir, "ion_network.gpkg")
    
    if verbose:
        print("\n" + "="*60)
        print("Building Multimodal ION Network")
        print("  - Bus routes from Raw_GTFS")
        print("  - ION LRT from GTFS(onlyLRT) + extension stops")
        print("  - Route shapes from GTFS shapes.txt")
        print("="*60)
    
    nodes_df, links_df = build_nodes_and_links(
        bus_gtfs,
        lrt_gtfs,
        ion_stops_csv,
        ion_routes_csv,
        taz_path=taz_path,
        assign_zone_id=assign_zone_id,
    )
    
    if verbose:
        print(f"\n=== Load route shapes from GTFS ===")
    link_shapes = {}
    if bus_gtfs:
        bus_shapes = build_link_shape_lookup(bus_gtfs, nodes_df, links_df, route_prefix="")
        link_shapes.update(bus_shapes)
        if verbose:
            print(f"  Bus: {len(bus_shapes)} route links with shapes")
    if lrt_gtfs:
        lrt_shapes = build_link_shape_lookup(lrt_gtfs, nodes_df, links_df, route_prefix=LRT_ROUTE_PREFIX)
        link_shapes.update(lrt_shapes)
        if verbose:
            print(f"  LRT: {len(lrt_shapes)} route links with shapes")

    # Extension links use straight-line geometry (stop coords only, no GeoJSON)

    if verbose:
        print(f"\n=== Save node.csv, link.csv, geometry.csv ===")
    save_network_data(nodes_df, links_df, link_shapes, data_dir, verbose=verbose)
    
    if export_arcgis:
        if verbose:
            print(f"\n=== Export to ArcGIS (from saved data) ===")
        export_to_arcgis_from_data(data_dir, gpkg_path, verbose=verbose)
    
    if export_aequilibrae:
        if verbose:
            print(f"\n=== Create AequilibraE Project ===")
        from .aequilibrae_network import create_aequilibrae_project
        create_aequilibrae_project(
            bus_gtfs=bus_gtfs,
            lrt_gtfs=lrt_gtfs,
            ion_stops_csv=ion_stops_csv,
            ion_routes_csv=ion_routes_csv,
            nodes_df=nodes_df,
            links_df=links_df,
            verbose=verbose,
        )
    
    if verbose:
        print("\n" + "="*60)
        print("Multimodal Network Build Complete!")
        print("="*60)
        print(f"  Total nodes: {len(nodes_df)}")
        print(f"    - Bus nodes: {len(nodes_df[nodes_df['mode'] == MODE_BUS])}")
        print(f"    - LRT nodes: {len(nodes_df[nodes_df['mode'] == MODE_LRT])}")
        print(f"  Total links: {len(links_df)}")
        print(f"\n  Cost parameters (for shortest path):")
        print(f"    - Fare: ${FARE_CONSTANT:.2f}")
        print(f"    - Value of time: ${VALUE_OF_TIME:.2f}/min")
        print(f"    - Waiting time: bus={WAITING_TIME_BUS:.1f} min, LRT={WAITING_TIME_LRT:.1f} min (by origin mode)")
    
    return nodes_df, links_df


if __name__ == "__main__":
    # Build the multimodal network
    nodes_df, links_df = build_network(export_arcgis=True, verbose=True)
    
    # Test shortest path and export to GeoPackage
    print("\n=== Testing Shortest Path ===")
    
    from transit.shortest_path import shortest_path_transit, compute_path_details, export_shortest_path_to_arcgis
    
    # Find sample LRT and bus nodes
    lrt_nodes = nodes_df[nodes_df["mode"] == MODE_LRT]
    bus_nodes = nodes_df[nodes_df["mode"] == MODE_BUS]
    
    if len(lrt_nodes) > 0 and len(bus_nodes) > 0:
        orig_node = int(1614)
        dest_node = int(2874)
        
        print(f"  Finding path from node {orig_node} to node {dest_node}")
        orig_mode = nodes_df[nodes_df["node_id"] == orig_node]["mode"].iloc[0] if orig_node in nodes_df["node_id"].values else MODE_BUS
        wait_min = WAITING_TIME_LRT if orig_mode == MODE_LRT else WAITING_TIME_BUS
        print(f"  Cost parameters: FARE=${FARE_CONSTANT:.2f}, VALUE_OF_TIME=${VALUE_OF_TIME:.2f}/min, Waiting time: {wait_min:.1f} min ({orig_mode})")
        
        result = shortest_path_transit(
            nodes_df, links_df,
            orig_node, dest_node,
            cost="generalized",
            fare=FARE_CONSTANT,
            waiting_time_min=wait_min,
            value_of_time=VALUE_OF_TIME,
            verbose=True
        )
        
        if result["found"]:
            print(f"\n  Path found!")
            print(f"    Nodes in path: {len(result['path_nodes'])}")
            print(f"    Total time: {result['total_time_min']:.1f} min")
            print(f"    Total distance: {result['total_length_m']:.0f} m")
            print(f"    Generalized cost: ${result['generalized_cost']:.2f}")
            
            details = compute_path_details(links_df, result["path_links"], nodes_df=nodes_df)
            print(f"\n  Path Statistics:")
            print(f"    Total distance: {details['total_length_m']:.0f} m")
            print(f"    Total travel time: {details['total_travel_time_min']:.1f} min")
            print(f"    Number of transfers: {details['num_transfers']}")
            print(f"    Routes used: {details['routes_used']}")
            
            # Build link shapes for proper route geometry in export
            link_shapes = {}
            bus_shapes = build_link_shape_lookup(BUS_GTFS_DIR, nodes_df, links_df, route_prefix="")
            link_shapes.update(bus_shapes)
            lrt_shapes = build_link_shape_lookup(LRT_GTFS_DIR, nodes_df, links_df, route_prefix=LRT_ROUTE_PREFIX)
            link_shapes.update(lrt_shapes)
            # Extension links use straight lines (no GeoJSON geometry)

            # Export shortest path to GeoPackage with route shapes
            print("\n=== Export Shortest Path to ArcGIS ===")
            arcgis_dir = os.path.join(os.path.dirname(__file__), "arcgis_export")
            export_shortest_path_to_arcgis(
                nodes_df, links_df,
                result["path_nodes"], result["path_links"],
                out_dir=arcgis_dir,
                link_shapes=link_shapes
            )
        else:
            print(f"  No path found")
    else:
        print("  Could not find test nodes")
