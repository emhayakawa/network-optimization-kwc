import geopandas as gpd  # use to read shapefiles, filter by road type, compute lengths
from shapely.geometry import LineString
from methods import *

# Speeds for time-equivalent link length (km/h). LRT is faster so gets shorter "cost" per meter.
BUS_SPEED_KMH = 20.0
LRT_SPEED_KMH = 35.0

def fill_aequilibrae_ba_columns(project, direction_columns=None):
    """
    Fill empty _ab and _ba columns (lanes, speed, capacity, travel_time) with 0 in the project DB.
    GMNS importer leaves one direction empty for one-way links; build_graphs requires both _ab and _ba to be numeric.
    """
    if direction_columns is None:
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

def _flow_direction_to_links(road, start_coord, end_coord, nodes, next_link_id):
    """
    Given FlowDirect in ['Two-Way', 'TwoWay', 'FromTo', 'ToFrom'], return a list of
    link dicts (one or two) and the next_link_id. Each link is directed (from->to only).
    """
    flow = str(road.get('FlowDirect', 'Two-Way')).strip()
    length = road.geometry.length
    link_type = road['CartoClass'].lower()
    name = road.get('StreetName', '')
    geom = road.geometry
    allowed_uses = 'car'
    lanes = road.get('NumberofLa', 1)

    def one_link(from_coord, to_coord):
        return {
            'link_id': next_link_id,
            'from_node_id': nodes[from_coord],
            'to_node_id': nodes[to_coord],
            'directed': 1,
            'length': length,
            'link_type': link_type,
            'name': name,
            'geometry': geom,
            'allowed_uses': allowed_uses,
            'lanes': lanes,
        }

    links = []
    if flow in ('Two-Way', 'TwoWay'):
        links.append(one_link(start_coord, end_coord))
        next_link_id += 1
        links.append(one_link(end_coord, start_coord))
        next_link_id += 1
    elif flow == 'FromTo':
        links.append(one_link(start_coord, end_coord))
        next_link_id += 1
    elif flow == 'ToFrom':
        links.append(one_link(end_coord, start_coord))
        next_link_id += 1
    else:
        # Unknown value: treat as two-way
        links.append(one_link(start_coord, end_coord))
        next_link_id += 1
        links.append(one_link(end_coord, start_coord))
        next_link_id += 1
    return links, next_link_id


def create_road_edges(roads_filtered, nodes, next_link_id):
    road_edges = []

    for idx, road in roads_filtered.iterrows():
        coords = list(road.geometry.coords)
        start_coord = snap(coords[0])
        end_coord = snap(coords[-1])

        if start_coord not in nodes or end_coord not in nodes:
            continue

        new_links, next_link_id = _flow_direction_to_links(
            road, start_coord, end_coord, nodes, next_link_id
        )
        road_edges.extend(new_links)

    road_edges_gdf = gpd.GeoDataFrame(road_edges, geometry='geometry', crs=roads_filtered.crs)
    return road_edges_gdf, next_link_id

def create_transit_edges(feed, filtered_stops_gdf, nodes, road_edges_gdf, roads_filtered, next_link_id, stop_to_representative):
    transit_edges = []
    seen_transit_pairs = set()
    overlap_tolerance = 10  # meters

    # Create spatial index for fast overlap checking
    road_sindex = road_edges_gdf.sindex

    for trip_id, trip_stops in feed.stop_times.groupby("trip_id"):
        trip_stops = trip_stops.sort_values("stop_sequence")
        stop_ids = trip_stops['stop_id'].tolist()
        
        for i in range(len(stop_ids) - 1):
            stop1_id = stop_ids[i]
            stop2_id = stop_ids[i + 1]
            
            # Map to representative stops
            stop1_id = stop_to_representative.get(stop1_id, stop1_id)
            stop2_id = stop_to_representative.get(stop2_id, stop2_id)

            if stop1_id not in filtered_stops_gdf.index or stop2_id not in filtered_stops_gdf.index:
                continue
            
            # Skip if we've already created this edge
            edge_pair = frozenset({stop1_id, stop2_id})
            if edge_pair in seen_transit_pairs:
                continue
            
            s1 = filtered_stops_gdf.loc[stop1_id]
            s2 = filtered_stops_gdf.loc[stop2_id]
            
            # Get node IDs from the snapped coordinates
            from_coord = (s1.geometry.x, s1.geometry.y)
            to_coord = (s2.geometry.x, s2.geometry.y)
            
            # Skip self-loops (same node)
            if from_coord == to_coord:
                seen_transit_pairs.add(edge_pair)
                continue
            
            from_node = nodes[from_coord]
            to_node = nodes[to_coord]
            edge_geom = LineString([s1.geometry, s2.geometry])
            
            # Check if transit edge overlaps with road edges using spatial index
            buffered = edge_geom.buffer(overlap_tolerance)
            possible_matches_idx = list(road_sindex.intersection(buffered.bounds))
            
            overlaps_road = False
            if possible_matches_idx:
                possible_matches = road_edges_gdf.iloc[possible_matches_idx]
                for _, road_edge in possible_matches.iterrows():
                    if road_edge.geometry.distance(edge_geom) <= overlap_tolerance:
                        overlaps_road = True
                        break
            
            if overlaps_road:
                seen_transit_pairs.add(edge_pair)
            else:
                length_m = s1.geometry.distance(s2.geometry)
                transit_edges.append({
                    'link_id': next_link_id,
                    'from_node_id': from_node,
                    'to_node_id': to_node,
                    'directed': 1,
                    'length': length_m,
                    'link_type': 'transit',
                    'route_id': str(trip_id),
                    'geometry': edge_geom,
                    'allowed_uses': 'transit',
                    'lanes': 1,
                })
                next_link_id += 1
                seen_transit_pairs.add(edge_pair)

    if transit_edges:
        transit_edges_gdf = gpd.GeoDataFrame(transit_edges, geometry='geometry', crs=roads_filtered.crs)
    else:
        transit_edges_gdf = gpd.GeoDataFrame(
            columns=['link_id', 'from_node_id', 'to_node_id', 'directed', 'length', 'link_type', 'route_id', 'geometry', 'allowed_uses', 'lanes'],
            geometry='geometry',
            crs=roads_filtered.crs
        )

    return transit_edges_gdf


def _ion_stop_coord(row):
    """Same rounding as add_ion_stops_to_nodes so (x,y) matches nodes dict."""
    g = row.geometry
    return (round(g.x, 3), round(g.y, 3))


def create_ion_edges(ion_stops_gdf, nodes, next_link_id, crs, bus_speed_kmh=BUS_SPEED_KMH, lrt_speed_kmh=LRT_SPEED_KMH):
    """
    Create GMNS links between consecutive ION LRT stops (both directions).
    Uses effective length = physical_length * (bus_speed / lrt_speed) so that shortest path
    in 'length' reflects faster travel on LRT than on bus.
    """
    # Order stops north to south (descending Y) to get stop sequence along the line
    ordered = ion_stops_gdf.sort_values(by="Y", ascending=False).reset_index(drop=True)
    ion_edges = []
    for i in range(len(ordered) - 1):
        r1, r2 = ordered.iloc[i], ordered.iloc[i + 1]
        c1, c2 = _ion_stop_coord(r1), _ion_stop_coord(r2)
        if c1 not in nodes or c2 not in nodes:
            continue
        from_node = nodes[c1]
        to_node = nodes[c2]
        geom = LineString([r1.geometry, r2.geometry])
        length_m = r1.geometry.distance(r2.geometry)
        # Time-equivalent length: LRT is faster, so lower cost per meter
        effective_length = length_m * (bus_speed_kmh / lrt_speed_kmh)
        # Forward direction (e.g. northbound)
        ion_edges.append({
            "link_id": next_link_id,
            "from_node_id": from_node,
            "to_node_id": to_node,
            "directed": 1,
            "length": effective_length,
            "link_type": "lrt",
            "geometry": geom,
            "allowed_uses": "transit",
            "lanes": 1,
        })
        next_link_id += 1
        # Reverse direction (e.g. southbound)
        ion_edges.append({
            "link_id": next_link_id,
            "from_node_id": to_node,
            "to_node_id": from_node,
            "directed": 1,
            "length": effective_length,
            "link_type": "lrt",
            "geometry": geom,
            "allowed_uses": "transit",
            "lanes": 1,
        })
        next_link_id += 1
    if not ion_edges:
        ion_gdf = gpd.GeoDataFrame(
            columns=["link_id", "from_node_id", "to_node_id", "directed", "length", "link_type", "geometry", "allowed_uses", "lanes"],
            geometry="geometry",
            crs=crs,
        )
    else:
        ion_gdf = gpd.GeoDataFrame(ion_edges, geometry="geometry", crs=crs)
    return ion_gdf, next_link_id

