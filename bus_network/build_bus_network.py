"""
Build a bus-only network from GTFS where:
- Nodes are (stop_id, route_id) so each physical stop has one node per route.
- Nearby stops (within STOP_CLUSTER_RADIUS_M) are clustered together and share coordinates.
- Links are either:
  1. Route segments: consecutive stops on the same route (travel time from GTFS, distance from coordinates).
     Each route is built in both directions (forward and reverse) so buses going either way are represented.
  2. Transfer links: same physical stop OR same cluster, different route — cost 15 minutes (configurable).
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
        DEFAULT_TAZ_SHAPEFILE,
        STOP_CLUSTER_RADIUS_M,
        FARE_CONSTANT,
        VALUE_OF_TIME,
        WAITING_TIME_MINUTES,
    )
    from zones import load_taz_zones, assign_zone_id_by_location
    from transit.shortest_path import shortest_path_transit, compute_path_details, export_shortest_path_to_arcgis
else:
    from .config import (
        TRANSFER_TIME_MINUTES,
        DEFAULT_GTFS_PATH,
        DEFAULT_TAZ_SHAPEFILE,
        STOP_CLUSTER_RADIUS_M,
        FARE_CONSTANT,
        VALUE_OF_TIME,
        WAITING_TIME_MINUTES,
    )
    from .zones import load_taz_zones, assign_zone_id_by_location
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


def build_nodes_and_links(gtfs_path=None, taz_path=None, assign_zone_id=True):
    """
    Build bus network nodes and links from GTFS.

    Nodes: one per (stop_id, route_id). Same physical stop has multiple nodes (one per route).
           Nearby stops (within STOP_CLUSTER_RADIUS_M) are clustered and share coordinates.
    Links:
      - Route segments: (stop_i, route) -> (stop_i+1, route) with travel_time_min and length_m.
      - Transfers: (stop, route_a) -> (stop, route_b) with travel_time_min = transfer_minutes, length_m = 0.
                   Also includes transfers between stops in the same cluster (nearby stops).

    If assign_zone_id is True and the TAZ file exists, each node gets zone_id from a point-in-polygon
    join at (stop_lat, stop_lon) (cluster coordinates).

    Returns:
      nodes_df: columns [node_id, stop_id, route_id, stop_lat, stop_lon, cluster_id] and zone_id if assigned
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

    # Add reverse direction only when the route actually runs both ways (reverse segment exists in trips)
    rev_links = []
    for _, row in route_links_df.iterrows():
        n1, n2 = row["from_node_id"], row["to_node_id"]
        s1 = nodes_df[nodes_df["node_id"] == n1]["stop_id"].iloc[0]
        s2 = nodes_df[nodes_df["node_id"] == n2]["stop_id"].iloc[0]
        route_id = row["route_id"]
        if (str(s2), str(s1), route_id) in segment_times:
            rev_links.append({
                "from_node_id": n2,
                "to_node_id": n1,
                "link_type": "route",
                "route_id": route_id,
                "travel_time_min": row["travel_time_min"],
                "length_m": row["length_m"],
            })
    if rev_links:
        rev_df = pd.DataFrame(rev_links)
        rev_df["link_id"] = np.arange(len(route_links_df) + 1, len(route_links_df) + len(rev_df) + 1)
        route_links_df["link_id"] = np.arange(1, len(route_links_df) + 1)
        route_links_df = pd.concat([route_links_df, rev_df], ignore_index=True)
    else:
        route_links_df["link_id"] = np.arange(1, len(route_links_df) + 1)

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

    # Concatenate route + transfer links
    links_df = pd.concat([route_links_df, transfer_links_df], ignore_index=True)

    if assign_zone_id:
        taz_file = taz_path or DEFAULT_TAZ_SHAPEFILE
        if os.path.isfile(taz_file):
            zones_gdf = load_taz_zones(taz_file)
            nodes_df = assign_zone_id_by_location(
                nodes_df, zones_gdf, lat_col="stop_lat", lon_col="stop_lon"
            )
        else:
            print(
                f"  TAZ layer not found ({taz_file}); "
                "skipping zone_id (set taz_path= or place shapefile under Data/...)"
            )

    # Clean up nodes_df columns for output
    node_out_cols = ["node_id", "stop_id", "route_id", "stop_lat", "stop_lon", "cluster_id"]
    if "zone_id" in nodes_df.columns:
        node_out_cols.append("zone_id")
    nodes_df = nodes_df[node_out_cols]
    
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


def save_network_data(nodes_df, links_df, link_shapes, out_dir, srid=26917, verbose=True):
    """
    Save network to node.csv, link.csv, geometry.csv (canonical output for ArcGIS and AequilibraE).
    
    Geometry uses link_shapes (GTFS route shapes) for route links, straight lines for others.
    Output is in project CRS (EPSG:26917) for consistency with road network.
    """
    import geopandas as gpd
    from shapely.geometry import LineString
    from shapely.ops import transform
    import pyproj

    os.makedirs(out_dir, exist_ok=True)
    
    # Nodes: transform to project CRS
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
    
    # Links: match links.csv schema (link_id, from_node_id, to_node_id, link_type, route_id, travel_time_min, length_m)
    link_cols = ["link_id", "from_node_id", "to_node_id", "link_type", "route_id", "travel_time_min", "length_m"]
    link_cols = [c for c in link_cols if c in links_df.columns]
    links_export = links_df[link_cols].copy()
    link_file = os.path.join(out_dir, "link.csv")
    links_export.to_csv(link_file, index=False)
    
    # Geometry: use link_shapes (route geometry) when available, else straight line
    # For transfer/other links: use display coords (x_coord, y_coord) so zero-length links get
    # visible geometry from the cluster offset; add min offset if still coincident
    node_xy = gdf_nodes.set_index("node_id")[["x_coord", "y_coord"]].to_dict("index")
    node_coord = nodes_df.set_index("node_id")[["stop_lat", "stop_lon"]].to_dict("index")
    transformer = pyproj.Transformer.from_crs("EPSG:4326", f"EPSG:{srid}", always_xy=True)
    TRANSFER_LINK_MIN_M = 2.0  # min visible length for zero-length links in display CRS
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
                # Ensure transfer-like links (near-zero length) are visible in ArcGIS
                if row.get("link_type") in ("transfer", "multimodal_transfer"):
                    dx, dy = x2 - x1, y2 - y1
                    d = (dx**2 + dy**2) ** 0.5
                    if d < TRANSFER_LINK_MIN_M:
                        if d < 1e-6:
                            x2, y2 = x1 + TRANSFER_LINK_MIN_M, y1  # east
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
    Transforms from project CRS to WGS84 for visualization.

    All columns present in ``node.csv`` (including ``zone_id`` when TAZ assignment ran) are
    written to the ``nodes`` layer — this is separate from ``bus_network/zones.export_zones_to_gis``,
    which only exports TAZ polygon boundaries for mapping.
    """
    import geopandas as gpd
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
    
    gdf_nodes = gpd.GeoDataFrame(
        nodes_df,
        geometry=gpd.points_from_xy(nodes_df["x_coord"], nodes_df["y_coord"]),
        crs="EPSG:26917",
    )
    gdf_nodes = gdf_nodes.to_crs("EPSG:4326")
    gdf_nodes["stop_lon"] = gdf_nodes.geometry.x
    gdf_nodes["stop_lat"] = gdf_nodes.geometry.y
    
    geom_lookup = geom_df.set_index("geometry_id")["geometry"].to_dict()
    link_geoms = []
    for _, row in links_df.iterrows():
        wkt_str = geom_lookup.get(row["link_id"])
        if wkt_str:
            link_geoms.append(wkt.loads(wkt_str))
        else:
            link_geoms.append(None)
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
    Uses trip-specific shapes: for each link, finds a trip that serves both stops
    in order, ensuring the correct shape is used when routes have multiple shapes
    (e.g., different directions or patterns).
    
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
    stop_times_file = os.path.join(path, "stop_times.txt")
    if not os.path.exists(trips_file):
        return {}
    
    trips_df = pd.read_csv(trips_file)
    trips_df["route_id"] = trips_df["route_id"].astype(str)
    trips_df["shape_id"] = trips_df["shape_id"].astype(str)
    
    # Build (from_stop, to_stop, route_id) -> shape_id from trips that serve both stops in order
    segment_to_shape = {}
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
    
    # Fallback: one shape per route (original behavior when no trip serves the segment)
    route_to_shape = {}
    for route_id, shape_id in trips_df.groupby("route_id")["shape_id"].first().items():
        route_to_shape[str(route_id)] = shape_id
    
    node_to_stop = nodes_df.set_index("node_id")["stop_id"].astype(str).to_dict()
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
        from_stop = node_to_stop.get(from_node)
        to_stop = node_to_stop.get(to_node)
        # Use trip-specific shape when available (handles routes with multiple shapes)
        shape_id = None
        if from_stop and to_stop:
            shape_id = segment_to_shape.get((from_stop, to_stop, route_id))
        if not shape_id:
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
            segment = list(coords[from_idx:to_idx + 1])
        else:
            segment = list(coords[to_idx:from_idx + 1][::-1])
        
        if len(segment) < 2:
            segment = [from_pt, to_pt]
        else:
            # Ensure link passes through nodes (stops may be offset from road centerline)
            segment[0] = from_pt
            segment[-1] = to_pt
        
        link_shapes[(from_node, to_node)] = segment
    
    # For reverse route links: use reversed forward segment when forward has proper route geometry.
    # This fixes reverse links that got straight-line geometry (2 points) when the shape didn't
    # cover that direction well.
    for _, link in route_links.iterrows():
        from_node = int(link["from_node_id"])
        to_node = int(link["to_node_id"])
        rev_segment = link_shapes.get((to_node, from_node))
        if rev_segment and len(rev_segment) >= 3:  # forward has real route geometry
            existing = link_shapes.get((from_node, to_node))
            if not existing or len(existing) < 3:  # replace straight-line (2 pts) with reversed
                link_shapes[(from_node, to_node)] = list(reversed(rev_segment))
    
    return link_shapes


def export_to_arcgis(nodes_df, links_df, out_path, format="gpkg", link_shapes=None):
    """
    Export bus network for use in ArcGIS (or QGIS).
    
    Prefer export_to_arcgis_from_data() to create gpkg from saved node/link/geometry.csv.
    This function is kept for compatibility when working with in-memory data.
    """
    if link_shapes is None:
        link_shapes = {}
    import geopandas as gpd
    from shapely.geometry import LineString, Point

    nodes_df = nodes_df.copy()
    links_df = links_df.copy()
    node_coord = nodes_df.set_index("node_id")[["stop_lat", "stop_lon"]].to_dict("index")

    gdf_nodes = gpd.GeoDataFrame(
        nodes_df,
        geometry=[Point(row["stop_lon"], row["stop_lat"]) for _, row in nodes_df.iterrows()],
        crs="EPSG:4326",
    )
    geoms = []
    for _, row in links_df.iterrows():
        f, t = int(row["from_node_id"]), int(row["to_node_id"])
        shape_coords = link_shapes.get((f, t))
        if shape_coords and len(shape_coords) >= 2:
            geoms.append(LineString(shape_coords))
        else:
            c1, c2 = node_coord.get(f), node_coord.get(t)
            if c1 and c2:
                geoms.append(LineString([(c1["stop_lon"], c1["stop_lat"]), (c2["stop_lon"], c2["stop_lat"])]))
            else:
                geoms.append(None)
    links_df = links_df.copy()
    links_df["geometry"] = geoms
    gdf_links = gpd.GeoDataFrame(links_df.dropna(subset=["geometry"]), geometry="geometry", crs="EPSG:4326")

    out_path = os.path.abspath(out_path)
    if not out_path.endswith(".gpkg"):
        out_path = os.path.join(out_path, "bus_network.gpkg") if os.path.isdir(out_path) else out_path + ".gpkg"
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    if os.path.exists(out_path):
        os.remove(out_path)
    gdf_nodes.to_file(out_path, layer="nodes", driver="GPKG")
    gdf_links.to_file(out_path, layer="links", driver="GPKG", mode="a")


def export_to_gmns(nodes_df, links_df, out_dir, link_shapes=None, srid=26917):
    """
    Export bus network to GMNS format for AequilibraE import.
    Writes node.csv, link.csv, geometry.csv. Uses link_shapes for route geometry when provided.
    """
    node_file, link_file, geometry_file = save_network_data(
        nodes_df, links_df, link_shapes or {}, out_dir, srid=srid, verbose=False
    )
    return node_file, link_file, geometry_file


def build_network(
    gtfs_path=None,
    taz_path=None,
    assign_zone_id=True,
    export_arcgis=True,
    verbose=True,
):
    """
    Main workflow: build the bus network from GTFS.
    
    Outputs node.csv, link.csv, geometry.csv to data/ (canonical format for ArcGIS and AequilibraE).
    ArcGIS gpkg is created FROM these saved files when export_arcgis=True.
    
    Parameters:
        gtfs_path: path to GTFS directory (default: Data/Raw_GTFS)
        taz_path: path to TAZ layer for zone_id on nodes (default: Data/.../2011 RMOW RTM TAZ_zone.shp)
        assign_zone_id: if True and taz_path exists, add zone_id to each node
        export_arcgis: create GeoPackage from saved data for visualization
        verbose: print progress
    
    Returns:
        nodes_df, links_df (in-memory for shortest path etc.)
    """
    if gtfs_path is None:
        gtfs_path = DEFAULT_GTFS_PATH
    
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    arcgis_dir = os.path.join(os.path.dirname(__file__), "arcgis_export")
    gpkg_path = os.path.join(arcgis_dir, "bus_network.gpkg")
    
    if verbose:
        print("\n=== Step 1: Load GTFS data ===")
        print(f"  GTFS path: {gtfs_path}")
    
    if verbose:
        print("\n=== Step 2: Build bus network nodes and links ===")
    nodes_df, links_df = build_nodes_and_links(
        gtfs_path, taz_path=taz_path, assign_zone_id=assign_zone_id
    )
    
    if verbose:
        print(f"\n  Network built: {len(nodes_df)} nodes, {len(links_df)} links")
        print(f"  Link types:")
        for link_type, count in links_df["link_type"].value_counts().items():
            print(f"    - {link_type}: {count}")
    
    if verbose:
        print("\n=== Step 3: Load route shapes from GTFS ===")
    link_shapes = build_link_shape_lookup(gtfs_path, nodes_df, links_df)
    if verbose:
        print(f"  Found shapes for {len(link_shapes)} route links")
    
    if verbose:
        print("\n=== Step 4: Save node.csv, link.csv, geometry.csv ===")
    save_network_data(nodes_df, links_df, link_shapes, data_dir, verbose=verbose)
    
    if export_arcgis:
        if verbose:
            print("\n=== Step 5: Export to ArcGIS (from saved data) ===")
        export_to_arcgis_from_data(data_dir, gpkg_path, verbose=verbose)
    
    if verbose:
        print("\n=== Bus network build complete! ===")
    
    return nodes_df, links_df


if __name__ == "__main__":
    gtfs_path = DEFAULT_GTFS_PATH
    
    nodes_df, links_df = build_network(
        gtfs_path=gtfs_path,
        export_arcgis=True,
        verbose=True
    )
    
    print("\n=== Testing shortest path (Generalized Cost) ===")
    all_node_ids = list(nodes_df['node_id'].values)
    orig_node = int(1614)
    dest_node = int(1600)
    
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
        
        details = compute_path_details(links_df, result["path_links"], nodes_df=nodes_df)
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
