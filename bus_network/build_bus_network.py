"""
Build a bus-only network from GTFS where:
- Nodes are (stop_id, route_id) so each physical stop has one node per route.
- Nearby stops (within STOP_CLUSTER_RADIUS_M) are clustered together and share coordinates.
- Links are either:
  1. Route segments: consecutive stops on the same route (travel time from GTFS, distance from coordinates).
     Each route is built in both directions (forward and reverse) so buses going either way are represented.
  2. Transfer links: same physical stop OR same cluster, different route — cost 15 minutes (configurable).
  3. Walking links: between physical stops within 200 m; travel_time = distance / walking speed (configurable).
"""
import os
import sys
import pandas as pd
import numpy as np

# Ensure URA root is in path for transit module (when imported as package)
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

# Handle both module import and direct execution
if __name__ == "__main__":
    _dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.dirname(_dir))
    sys.path.insert(0, _dir)
    from config import (
        TRANSFER_TIME_MINUTES,
        DEFAULT_GTFS_PATH,
        WALKING_LINK_MAX_M,
        WALKING_SPEED_M_PER_MIN,
        STOP_CLUSTER_RADIUS_M,
        FARE_CONSTANT,
        VALUE_OF_TIME,
        WAITING_TIME_MINUTES,
    )
    from transit.shortest_path import shortest_path_transit, compute_path_details, export_shortest_path_to_arcgis
else:
    from .config import (
        TRANSFER_TIME_MINUTES,
        DEFAULT_GTFS_PATH,
        WALKING_LINK_MAX_M,
        WALKING_SPEED_M_PER_MIN,
        STOP_CLUSTER_RADIUS_M,
        FARE_CONSTANT,
        VALUE_OF_TIME,
        WAITING_TIME_MINUTES,
    )
    from transit.shortest_path import shortest_path_transit, compute_path_details, export_shortest_path_to_arcgis


def _parse_time(s):
    """Parse GTFS time 'H:MM:SS' or 'HH:MM:SS' to minutes since midnight."""
    if pd.isna(s) or s == "":
        return np.nan
    parts = str(s).strip().split(":")
    if len(parts) < 3:
        return np.nan
    h, m, sec = int(parts[0]), int(parts[1]), int(parts[2])
    return h * 60 + m + sec / 60.0


def _haversine_m(lat1, lon1, lat2, lon2):
    """Approximate distance in meters between two WGS84 points."""
    R = 6371000  # Earth radius in m
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c


def load_gtfs(gtfs_path=None):
    """Load routes, stops, stop_times, trips from GTFS directory (Raw_GTFS). Returns dict of DataFrames."""
    path = gtfs_path or DEFAULT_GTFS_PATH
    data = {}
    for name in ["routes", "stops", "stop_times", "trips"]:
        fname = os.path.join(path, name + ".txt")
        if not os.path.exists(fname):
            raise FileNotFoundError(f"GTFS file not found: {fname}")
        data[name] = pd.read_csv(fname)
    return data


def cluster_nearby_stops(stops_df, cluster_radius_m=None):
    """
    Cluster nearby bus stops within cluster_radius_m of each other.
    
    Uses a greedy approach: iterate through stops, if a stop is within radius
    of an existing cluster centroid, add it to that cluster. Otherwise create a new cluster.
    
    Returns:
        stops_df with additional columns:
        - cluster_id: identifier for the cluster this stop belongs to
        - cluster_lat: latitude of the cluster centroid
        - cluster_lon: longitude of the cluster centroid
    """
    if cluster_radius_m is None:
        cluster_radius_m = STOP_CLUSTER_RADIUS_M
    
    stops = stops_df.copy()
    stops["stop_id"] = stops["stop_id"].astype(str)
    
    # Initialize cluster assignments
    stops["cluster_id"] = -1
    stops["cluster_lat"] = np.nan
    stops["cluster_lon"] = np.nan
    
    clusters = []  # List of (cluster_id, centroid_lat, centroid_lon, [stop_indices])
    next_cluster_id = 0
    
    for idx, row in stops.iterrows():
        lat, lon = float(row["stop_lat"]), float(row["stop_lon"])
        assigned = False
        
        # Check if this stop is close enough to any existing cluster
        for cluster in clusters:
            c_id, c_lat, c_lon, members = cluster
            dist = _haversine_m(lat, lon, c_lat, c_lon)
            if dist <= cluster_radius_m:
                # Add to this cluster
                members.append(idx)
                # Update centroid (average of all member coordinates)
                member_lats = stops.loc[members, "stop_lat"].astype(float).values
                member_lons = stops.loc[members, "stop_lon"].astype(float).values
                cluster[1] = np.mean(member_lats)
                cluster[2] = np.mean(member_lons)
                assigned = True
                break
        
        if not assigned:
            # Create new cluster
            clusters.append([next_cluster_id, lat, lon, [idx]])
            next_cluster_id += 1
    
    # Assign cluster info to stops dataframe
    for cluster in clusters:
        c_id, c_lat, c_lon, members = cluster
        for idx in members:
            stops.loc[idx, "cluster_id"] = c_id
            stops.loc[idx, "cluster_lat"] = c_lat
            stops.loc[idx, "cluster_lon"] = c_lon
    
    n_clustered = sum(1 for c in clusters if len(c[3]) > 1)
    n_stops_in_clusters = sum(len(c[3]) for c in clusters if len(c[3]) > 1)
    print(f"Stop clustering: {n_clustered} clusters formed from {n_stops_in_clusters} stops "
          f"(radius={cluster_radius_m}m)")
    
    return stops


def build_nodes_and_links(gtfs_path=None):
    """
    Build bus network nodes and links from GTFS.

    Nodes: one per (stop_id, route_id). Same physical stop has multiple nodes (one per route).
           Nearby stops (within STOP_CLUSTER_RADIUS_M) are clustered and share coordinates.
    Links:
      - Route segments: (stop_i, route) -> (stop_i+1, route) with travel_time_min and length_m.
      - Transfers: (stop, route_a) -> (stop, route_b) with travel_time_min = transfer_minutes, length_m = 0.
                   Also includes transfers between stops in the same cluster (nearby stops).
      - Walking: between nodes at different stops within WALKING_LINK_MAX_M; travel_time_min = length_m / WALKING_SPEED_M_PER_MIN.

    Returns:
      nodes_df: columns [node_id, stop_id, route_id, stop_lat, stop_lon, cluster_id]
      links_df: columns [link_id, from_node_id, to_node_id, link_type, travel_time_min, length_m]
    """
    transfer_minutes = TRANSFER_TIME_MINUTES
    data = load_gtfs(gtfs_path)
    stops = data["stops"]
    stop_times = data["stop_times"]
    trips = data["trips"]

    # Ensure stop_id types match for merges
    stops["stop_id"] = stops["stop_id"].astype(str)
    stop_times["stop_id"] = stop_times["stop_id"].astype(str)
    trips["route_id"] = trips["route_id"].astype(str)

    # Cluster nearby stops
    stops = cluster_nearby_stops(stops, STOP_CLUSTER_RADIUS_M)

    # Join stop_times with trips to get (trip_id, stop_id, stop_sequence, route_id)
    st = stop_times.merge(trips[["trip_id", "route_id"]], on="trip_id", how="left")
    st = st.sort_values(["trip_id", "stop_sequence"])

    # Unique (stop_id, route_id) pairs
    stop_route_pairs = st[["stop_id", "route_id"]].drop_duplicates()
    stop_route_pairs = stop_route_pairs.reset_index(drop=True)
    stop_route_pairs["node_id"] = np.arange(1, len(stop_route_pairs) + 1)

    sr_to_node = {}
    for _, row in stop_route_pairs.iterrows():
        sr_to_node[(row["stop_id"], row["route_id"])] = int(row["node_id"])

    # Merge stop coordinates and cluster info into stop_route_pairs for nodes output
    # Use cluster coordinates instead of original stop coordinates
    stops_coords = stops[["stop_id", "stop_lat", "stop_lon", "cluster_id", "cluster_lat", "cluster_lon"]].drop_duplicates()
    nodes_df = stop_route_pairs.merge(stops_coords, on="stop_id", how="left")
    
    # Use cluster coordinates for node positions (so clustered stops share same location)
    # Keep original stop coordinates for reference before overwriting
    nodes_df["original_stop_lat"] = nodes_df["stop_lat"]
    nodes_df["original_stop_lon"] = nodes_df["stop_lon"]
    # Use cluster coordinates as the main stop_lat/stop_lon
    nodes_df["stop_lat"] = nodes_df["cluster_lat"]
    nodes_df["stop_lon"] = nodes_df["cluster_lon"]
    # Select final columns
    nodes_df = nodes_df[["node_id", "stop_id", "route_id", "stop_lat", "stop_lon", 
                         "cluster_id", "original_stop_lat", "original_stop_lon"]]

    # --- Route segment links (consecutive stops on same route) ---
    # Use ALL trips to capture all stop patterns (short-turns, express, etc.)
    # This ensures we don't miss connections that only appear in some trips
    st["arrival_min"] = st["arrival_time"].map(_parse_time)
    st["departure_min"] = st["departure_time"].map(_parse_time)

    # Collect all unique (from_stop, to_stop, route) segments with travel times
    segment_times = {}  # (from_stop, to_stop, route) -> list of travel times
    
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
        
        # Distance: use original stops coordinates (not clustered)
        r1 = stops[stops["stop_id"] == s1].iloc[0]
        r2 = stops[stops["stop_id"] == s2].iloc[0]
        lat1, lon1 = float(r1["stop_lat"]), float(r1["stop_lon"])
        lat2, lon2 = float(r2["stop_lat"]), float(r2["stop_lon"])
        length_m = _haversine_m(lat1, lon1, lat2, lon2)
        
        route_links.append({
            "link_id": link_id,
            "from_node_id": n1,
            "to_node_id": n2,
            "link_type": "route",
            "route_id": route_id,
            "travel_time_min": round(avg_travel_min, 2),
            "length_m": round(length_m, 2),
        })
        link_id += 1

    # Create DataFrame (already unique by construction)
    route_links_df = pd.DataFrame(route_links)

    # Add reverse direction for each route segment (bus can travel both ways along route)
    rev = route_links_df.rename(columns={"from_node_id": "to_node_id", "to_node_id": "from_node_id"})
    rev = rev[["from_node_id", "to_node_id", "link_type", "route_id", "travel_time_min", "length_m"]]
    rev["link_id"] = np.arange(len(route_links_df) + 1, len(route_links_df) + len(rev) + 1)
    route_links_df["link_id"] = np.arange(1, len(route_links_df) + 1)
    route_links_df = pd.concat([route_links_df, rev], ignore_index=True)

    # --- Transfer links (same stop_id OR same cluster_id, different route_id) ---
    # Build a mapping from cluster_id to list of node_ids
    cluster_to_nodes = nodes_df.groupby("cluster_id")["node_id"].apply(lambda x: x.astype(int).tolist()).to_dict()
    
    transfer_links = []
    link_id = len(route_links_df) + 1
    
    # For each cluster, link every pair of nodes (allowing transfers between different routes)
    for cluster_id, nids in cluster_to_nodes.items():
        if len(nids) <= 1:
            continue
        # Get route_id for each node to avoid same-route transfers
        node_routes = nodes_df.set_index("node_id")["route_id"].to_dict()
        for i, n_from in enumerate(nids):
            for j, n_to in enumerate(nids):
                if i != j and node_routes.get(n_from) != node_routes.get(n_to):
                    transfer_links.append({
                        "link_id": link_id,
                        "from_node_id": n_from,
                        "to_node_id": n_to,
                        "link_type": "transfer",
                        "route_id": None,
                        "travel_time_min": transfer_minutes,
                        "length_m": 0.0,
                    })
                    link_id += 1

    transfer_links_df = pd.DataFrame(transfer_links)
    if len(transfer_links_df) > 0:
        transfer_links_df["link_id"] = np.arange(len(route_links_df) + 1, len(route_links_df) + len(transfer_links_df) + 1)

    # --- Walking links: between physical stops within WALKING_LINK_MAX_M ---
    # NOTE: Temporarily disabled (user request). Set `enable_walking_links = True` to re-enable.
    enable_walking_links = False
    walking_links_df = pd.DataFrame()
    if enable_walking_links:
        # travel_time_min = distance_m / WALKING_SPEED_M_PER_MIN (walking minutes)
        stop_coords = stops[["stop_id", "stop_lat", "stop_lon"]].drop_duplicates()
        stop_to_nodes = nodes_df.groupby("stop_id")["node_id"].apply(lambda x: x.astype(int).tolist()).to_dict()
        walking_links = []
        next_link_id = len(route_links_df) + len(transfer_links_df) + 1
        for idx_i in range(len(stop_coords)):
            for idx_j in range(idx_i + 1, len(stop_coords)):
                ri, rj = stop_coords.iloc[idx_i], stop_coords.iloc[idx_j]
                d_m = _haversine_m(
                    float(ri["stop_lat"]), float(ri["stop_lon"]),
                    float(rj["stop_lat"]), float(rj["stop_lon"]),
                )
                if d_m > WALKING_LINK_MAX_M:
                    continue
                nids_i = stop_to_nodes.get(str(ri["stop_id"]), [])
                nids_j = stop_to_nodes.get(str(rj["stop_id"]), [])
                if not nids_i or not nids_j:
                    continue
                walk_min = d_m / WALKING_SPEED_M_PER_MIN
                for n_i in nids_i:
                    for n_j in nids_j:
                        walking_links.append({
                            "link_id": next_link_id,
                            "from_node_id": n_i,
                            "to_node_id": n_j,
                            "link_type": "walk",
                            "route_id": None,
                            "travel_time_min": round(walk_min, 2),
                            "length_m": round(d_m, 2),
                        })
                        next_link_id += 1
                        walking_links.append({
                            "link_id": next_link_id,
                            "from_node_id": n_j,
                            "to_node_id": n_i,
                            "link_type": "walk",
                            "route_id": None,
                            "travel_time_min": round(walk_min, 2),
                            "length_m": round(d_m, 2),
                        })
                        next_link_id += 1
        walking_links_df = pd.DataFrame(walking_links)

    # Concatenate route + transfer (+ optional walking) links
    parts = [route_links_df, transfer_links_df]
    if len(walking_links_df) > 0:
        parts.append(walking_links_df)
    links_df = pd.concat(parts, ignore_index=True)
    
    # Clean up nodes_df columns for output
    nodes_df = nodes_df[["node_id", "stop_id", "route_id", "stop_lat", "stop_lon", "cluster_id"]]
    
    # Diagnostic: check for disconnected nodes
    all_node_ids = set(nodes_df["node_id"].astype(int))
    connected_from = set(links_df["from_node_id"].astype(int))
    connected_to = set(links_df["to_node_id"].astype(int))
    connected_nodes = connected_from | connected_to
    disconnected = all_node_ids - connected_nodes
    
    if disconnected:
        print(f"  WARNING: {len(disconnected)} nodes have no links (disconnected)")
        # Show some examples
        disconnected_df = nodes_df[nodes_df["node_id"].isin(disconnected)]
        routes_affected = disconnected_df["route_id"].unique()
        print(f"    Routes with disconnected nodes: {len(routes_affected)}")
        if len(routes_affected) <= 10:
            print(f"    Route IDs: {list(routes_affected)}")
    else:
        print(f"  All {len(all_node_ids)} nodes are connected")

    return nodes_df, links_df


def save_network(nodes_df, links_df, out_dir, verbose=True):
    """Save nodes and links to CSV in out_dir."""
    os.makedirs(out_dir, exist_ok=True)
    nodes_path = os.path.join(out_dir, "nodes.csv")
    links_path = os.path.join(out_dir, "links.csv")
    nodes_df.to_csv(nodes_path, index=False)
    links_df.to_csv(links_path, index=False)
    if verbose:
        print(f"  Nodes saved to: {nodes_path}")
        print(f"  Links saved to: {links_path}")


def load_gtfs_shapes(gtfs_path=None):
    """
    Load shapes.txt from GTFS directory (Raw_GTFS).
    
    Returns:
        dict: shape_id -> list of (lon, lat) coordinates in sequence order
    """
    path = gtfs_path or DEFAULT_GTFS_PATH
    shapes_file = os.path.join(path, "shapes.txt")
    if not os.path.exists(shapes_file):
        return {}
    shapes_df = pd.read_csv(shapes_file)
    shapes_df = shapes_df.sort_values(["shape_id", "shape_pt_sequence"])
    
    shape_coords = {}
    for shape_id, grp in shapes_df.groupby("shape_id"):
        coords = list(zip(grp["shape_pt_lon"].values, grp["shape_pt_lat"].values))
        shape_coords[str(shape_id)] = coords
    
    return shape_coords


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


def build_link_shape_lookup(gtfs_path, nodes_df, links_df):
    """
    Build a lookup from (from_node_id, to_node_id) to shape geometry.
    
    For each route link, finds the segment of the shape between the two stops.
    
    Parameters:
        gtfs_path: Path to GTFS directory (Raw_GTFS)
        nodes_df: Network nodes
        links_df: Network links
    
    Returns:
        dict: (from_node_id, to_node_id) -> list of (lon, lat) coordinates
    """
    shape_coords = load_gtfs_shapes(gtfs_path)
    if not shape_coords:
        return {}
    
    path = gtfs_path or DEFAULT_GTFS_PATH
    trips_file = os.path.join(path, "trips.txt")
    if not os.path.exists(trips_file):
        return {}
    trips_df = pd.read_csv(trips_file)
    
    trips_df["route_id"] = trips_df["route_id"].astype(str)
    trips_df["shape_id"] = trips_df["shape_id"].astype(str)
    
    route_to_shape = {}
    for route_id, shape_id in trips_df.groupby("route_id")["shape_id"].first().items():
        route_to_shape[str(route_id)] = shape_id
    
    node_coords = nodes_df.set_index("node_id")[["stop_lat", "stop_lon"]].to_dict("index")
    
    link_shapes = {}
    route_links = links_df[links_df["link_type"] == "route"].copy()
    
    for _, link in route_links.iterrows():
        from_node = int(link["from_node_id"])
        to_node = int(link["to_node_id"])
        route_id = link.get("route_id")
        
        if pd.isna(route_id):
            continue
        
        route_id = str(route_id)
        shape_id = route_to_shape.get(route_id)
        
        if not shape_id or shape_id not in shape_coords:
            continue
        
        coords = shape_coords[shape_id]
        if len(coords) < 2:
            continue
        
        c1 = node_coords.get(from_node)
        c2 = node_coords.get(to_node)
        
        if not c1 or not c2:
            continue
        
        from_pt = (float(c1["stop_lon"]), float(c1["stop_lat"]))
        to_pt = (float(c2["stop_lon"]), float(c2["stop_lat"]))
        
        from_idx = _find_closest_point_idx(coords, from_pt)
        to_idx = _find_closest_point_idx(coords, to_pt)
        
        if from_idx <= to_idx:
            segment = coords[from_idx:to_idx + 1]
        else:
            segment = coords[to_idx:from_idx + 1][::-1]
        
        if len(segment) < 2:
            segment = [from_pt, to_pt]
        
        link_shapes[(from_node, to_node)] = segment
    
    return link_shapes


def export_to_arcgis(nodes_df, links_df, out_path, format="gpkg", link_shapes=None):
    """
    Export bus network for use in ArcGIS (or QGIS).

    Parameters
    ----------
    nodes_df : pandas.DataFrame
        From build_nodes_and_links (columns: node_id, stop_id, route_id, stop_lat, stop_lon).
    links_df : pandas.DataFrame
        From build_nodes_and_links (columns: link_id, from_node_id, to_node_id, link_type, route_id, travel_time_min, length_m).
    out_path : str
        Output path: path to a .gpkg file (one file with layers 'nodes' and 'links'),
        or a directory for shapefiles (writes bus_network_nodes.shp, bus_network_links.shp).
    format : str
        'gpkg' (default) or 'shp'. For 'gpkg', use a path ending in .gpkg or a folder (creates bus_network.gpkg inside).
    link_shapes : dict, optional
        (from_node_id, to_node_id) -> list of (lon, lat) coords from GTFS shapes.txt
    """
    import geopandas as gpd
    from shapely.geometry import LineString, Point

    nodes_df = nodes_df.copy()
    links_df = links_df.copy()
    
    if link_shapes is None:
        link_shapes = {}
    
    node_coord = nodes_df.set_index("node_id")[["stop_lat", "stop_lon"]].to_dict("index")

    gdf_nodes = gpd.GeoDataFrame(
        nodes_df,
        geometry=[Point(row["stop_lon"], row["stop_lat"]) for _, row in nodes_df.iterrows()],
        crs="EPSG:4326",
    )

    geoms = []
    n_with_shapes = 0
    for _, row in links_df.iterrows():
        f, t = int(row["from_node_id"]), int(row["to_node_id"])
        
        shape_coords = link_shapes.get((f, t))
        
        if shape_coords and len(shape_coords) >= 2:
            geoms.append(LineString(shape_coords))
            n_with_shapes += 1
        else:
            c1 = node_coord.get(f)
            c2 = node_coord.get(t)
            if c1 and c2:
                geoms.append(LineString([(c1["stop_lon"], c1["stop_lat"]), (c2["stop_lon"], c2["stop_lat"])]))
            else:
                geoms.append(None)
    
    links_df = links_df.copy()
    links_df["geometry"] = geoms
    gdf_links = gpd.GeoDataFrame(links_df.dropna(subset=["geometry"]), geometry="geometry", crs="EPSG:4326")

    out_path = os.path.abspath(out_path)
    if out_path.endswith(".gpkg") or format == "gpkg":
        if not out_path.endswith(".gpkg"):
            out_path = os.path.join(out_path, "bus_network.gpkg") if os.path.isdir(out_path) else out_path + ".gpkg"
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        if os.path.exists(out_path):
            os.remove(out_path)
        gdf_nodes.to_file(out_path, layer="nodes", driver="GPKG")
        gdf_links.to_file(out_path, layer="links", driver="GPKG", mode="a")
    else:
        os.makedirs(out_path, exist_ok=True)
        nodes_shp = os.path.join(out_path, "bus_network_nodes.shp")
        links_shp = os.path.join(out_path, "bus_network_links.shp")
        gdf_nodes.to_file(nodes_shp, driver="ESRI Shapefile")
        gdf_links.to_file(links_shp, driver="ESRI Shapefile")
    
    print(f"    - Links with route shapes: {n_with_shapes}")
    print(f"    - Links with straight lines: {len(gdf_links) - n_with_shapes}")


def export_to_gmns(nodes_df, links_df, out_dir, srid=26917):
    """
    Export bus network to GMNS format for AequilibraE import.
    Writes node.csv (node_id, x_coord, y_coord), link.csv, and geometry.csv.
    Node coordinates are transformed from WGS84 (stop_lat, stop_lon) to the given SRID (default 26917).
    """
    import geopandas as gpd
    from shapely.geometry import LineString

    os.makedirs(out_dir, exist_ok=True)
    # Transform node coordinates from WGS84 to project CRS
    gdf_nodes = gpd.GeoDataFrame(
        nodes_df,
        geometry=gpd.points_from_xy(nodes_df["stop_lon"], nodes_df["stop_lat"]),
        crs="EPSG:4326",
    )
    gdf_nodes = gdf_nodes.to_crs(f"EPSG:{srid}")
    gdf_nodes["x_coord"] = gdf_nodes.geometry.x
    gdf_nodes["y_coord"] = gdf_nodes.geometry.y
    # AequilibraE rejects multiple nodes at same location. Offset only duplicate nodes (same cluster).
    # Group by cluster_id and offset nodes within same cluster
    if "cluster_id" in gdf_nodes.columns:
        delta_m = 0.5
        for cluster_id, idxs in gdf_nodes.groupby("cluster_id").groups.items():
            if len(idxs) <= 1:
                continue
            # Multiple nodes in same cluster: give each a small offset so they don't overlap
            for i, idx in enumerate(idxs):
                gdf_nodes.loc[idx, "x_coord"] = gdf_nodes.loc[idx, "x_coord"] + (i % 10) * delta_m
                gdf_nodes.loc[idx, "y_coord"] = gdf_nodes.loc[idx, "y_coord"] + (i // 10) * delta_m
    else:
        # Fallback: offset by lat/lon (old behavior)
        stop_key = (
            gdf_nodes["stop_lat"].round(6).astype(str) + "_" + gdf_nodes["stop_lon"].round(6).astype(str)
        )
        delta_m = 0.5
        for _, idxs in stop_key.groupby(stop_key).groups.items():
            if len(idxs) <= 1:
                continue
            for i, idx in enumerate(idxs):
                gdf_nodes.loc[idx, "x_coord"] = gdf_nodes.loc[idx, "x_coord"] + (i % 10) * delta_m
                gdf_nodes.loc[idx, "y_coord"] = gdf_nodes.loc[idx, "y_coord"] + (i // 10) * delta_m
    node_file = os.path.join(out_dir, "node.csv")
    gdf_nodes[["node_id", "x_coord", "y_coord"]].to_csv(node_file, index=False)

    # Build node_id -> (x, y) for link geometry
    coord = gdf_nodes.set_index("node_id")[["x_coord", "y_coord"]].to_dict("index")

    # GMNS links: link_id, from_node_id, to_node_id, directed, length, geometry_id, allowed_uses, lanes, travel_time
    links_export = links_df[["link_id", "from_node_id", "to_node_id", "length_m", "travel_time_min"]].copy()
    links_export["directed"] = 1
    links_export["geometry_id"] = links_export["link_id"]
    links_export["allowed_uses"] = "transit"
    links_export["lanes"] = 1
    links_export["length"] = links_export["length_m"]
    # Export travel time in minutes for use as cost in AequilibraE
    links_export["travel_time"] = links_export["travel_time_min"]
    link_file = os.path.join(out_dir, "link.csv")
    links_export[
        [
            "link_id",
            "from_node_id",
            "to_node_id",
            "directed",
            "length",
            "geometry_id",
            "allowed_uses",
            "lanes",
            "travel_time",
        ]
    ].to_csv(link_file, index=False)

    # Geometry: geometry_id, geometry (WKT)
    geoms = []
    for _, row in links_df.iterrows():
        f, t = int(row["from_node_id"]), int(row["to_node_id"])
        pf = coord.get(f, {})
        pt = coord.get(t, {})
        if pf and pt:
            line = LineString([(pf["x_coord"], pf["y_coord"]), (pt["x_coord"], pt["y_coord"])])
            geoms.append({"geometry_id": row["link_id"], "geometry": line.wkt})
    geom_df = pd.DataFrame(geoms)
    geometry_file = os.path.join(out_dir, "geometry.csv")
    geom_df.to_csv(geometry_file, index=False)
    return node_file, link_file, geometry_file


def build_network(
    gtfs_path=None,
    export_arcgis=True,
    export_gmns=False,
    verbose=True
):
    """
    Main workflow: build the bus network from GTFS.
    
    Parameters:
        gtfs_path: path to GTFS directory (default: Data/Raw_GTFS)
        export_arcgis: export to GeoPackage for visualization
        export_gmns: export to GMNS format for AequilibraE
        verbose: print progress
    
    Returns:
        nodes_df: DataFrame of nodes (stop_id, route_id, coordinates, cluster_id)
        links_df: DataFrame of links (route segments, transfers, walking links)
    """
    if gtfs_path is None:
        gtfs_path = DEFAULT_GTFS_PATH
    
    if verbose:
        print("\n=== Step 1: Load GTFS data ===")
        print(f"  GTFS path: {gtfs_path}")
    
    if verbose:
        print("\n=== Step 2: Build bus network nodes and links ===")
    nodes_df, links_df = build_nodes_and_links(gtfs_path)
    
    if verbose:
        print(f"\n  Network built: {len(nodes_df)} nodes, {len(links_df)} links")
        print(f"  Link types:")
        for link_type, count in links_df["link_type"].value_counts().items():
            print(f"    - {link_type}: {count}")
    
    if verbose:
        print("\n=== Step 3: Save network to CSV ===")
    out_dir = os.path.join(os.path.dirname(__file__), "data")
    save_network(nodes_df, links_df, out_dir, verbose=verbose)
    
    # Build link shapes from GTFS shapes.txt for proper route geometry
    link_shapes = {}
    if export_arcgis:
        if verbose:
            print("\n=== Step 4: Load route shapes from GTFS ===")
        link_shapes = build_link_shape_lookup(gtfs_path, nodes_df, links_df)
        if verbose:
            print(f"  Found shapes for {len(link_shapes)} route links")
    
    if export_arcgis:
        if verbose:
            print("\n=== Step 5: Export to ArcGIS (GeoPackage with route shapes) ===")
        arcgis_dir = os.path.join(os.path.dirname(__file__), "arcgis_export")
        gpkg_path = os.path.join(arcgis_dir, "bus_network.gpkg")
        export_to_arcgis(nodes_df, links_df, gpkg_path, format="gpkg", link_shapes=link_shapes)
        if verbose:
            print(f"  ArcGIS export saved to: {gpkg_path}")
            print("    - Layer 'nodes': bus stop nodes (Point)")
            print("    - Layer 'links': route segments, transfers (LineString)")
    
    if export_gmns:
        if verbose:
            print("\n=== Step 6: Export to GMNS format ===")
        gmns_dir = os.path.join(os.path.dirname(__file__), "gmns_export")
        node_file, link_file, geometry_file = export_to_gmns(nodes_df, links_df, gmns_dir)
        if verbose:
            print(f"  GMNS export saved to: {gmns_dir}")
            print(f"    - node.csv: {node_file}")
            print(f"    - link.csv: {link_file}")
            print(f"    - geometry.csv: {geometry_file}")
    
    if verbose:
        print("\n=== Bus network build complete! ===")
    
    return nodes_df, links_df


if __name__ == "__main__":
    gtfs_path = DEFAULT_GTFS_PATH
    
    nodes_df, links_df = build_network(
        gtfs_path=gtfs_path,
        export_arcgis=True,
        export_gmns=False,
        verbose=True
    )
    
    print("\n=== Testing shortest path (Generalized Cost) ===")
    all_node_ids = list(nodes_df['node_id'].values)
    orig_node = int(all_node_ids[0])
    dest_node = int(all_node_ids[min(100, len(all_node_ids) - 1)])
    
    print(f"  Finding path from node {orig_node} to node {dest_node}")
    print(f"  Cost parameters: FARE=${FARE_CONSTANT:.2f}, VALUE_OF_TIME=${VALUE_OF_TIME:.2f}/min")
    
    result = shortest_path_transit(
        nodes_df, links_df, orig_node, dest_node,
        cost="generalized",
        fare=FARE_CONSTANT,
        waiting_time_min=WAITING_TIME_MINUTES,
        value_of_time=VALUE_OF_TIME,
        verbose=True
    )
    
    if result.get("found"):
        print(f"\n  Path found!")
        print(f"    Nodes in path: {len(result['path_nodes'])}")
        print(f"    Total time: {result['total_time_min']:.1f} min")
        print(f"    Total distance: {result['total_length_m']:.0f} m")
        print(f"    Generalized cost: ${result['generalized_cost']:.2f}")
        
        details = compute_path_details(links_df, result["path_links"])
        print(f"\n  Path Statistics:")
        print(f"    Total distance: {details['total_length_m']:.0f} m")
        print(f"    Total travel time: {details['total_travel_time_min']:.1f} min")
        print(f"    Number of transfers: {details['num_transfers']}")
        print(f"    Routes used: {details['routes_used']}")
        
        # Build link shapes for shortest path export
        print("\n=== Building link shapes for shortest path export ===")
        link_shapes = build_link_shape_lookup(gtfs_path, nodes_df, links_df)
        print(f"  Found shapes for {len(link_shapes)} route links")
        
        print("\n=== Export shortest path to ArcGIS ===")
        arcgis_dir = os.path.join(os.path.dirname(__file__), "arcgis_export")
        export_shortest_path_to_arcgis(
            nodes_df, links_df,
            result["path_nodes"], result["path_links"],
            out_dir=arcgis_dir,
            link_shapes=link_shapes
        )
    else:
        print(f"  No path found")
