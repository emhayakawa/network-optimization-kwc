"""
Link/edge creation functions for the road network.
"""
import geopandas as gpd
import numpy as np
from shapely.geometry import LineString

from .nodes import snap
from .config import SPEED_FIELD, DEFAULT_SPEED_KMH


def _adjust_geometry_endpoints(geom, new_start, new_end):
    """
    Adjust a LineString geometry so its endpoints match the clustered node locations.
    
    Only modifies the first and last coordinates; the intermediate road path is preserved.
    
    Parameters:
        geom: LineString geometry
        new_start: (x, y) tuple for the start point (clustered node location)
        new_end: (x, y) tuple for the end point (clustered node location)
    
    Returns:
        Adjusted LineString geometry
    """
    coords = list(geom.coords)
    coords[0] = new_start
    coords[-1] = new_end
    return LineString(coords)


def _get_speed_kmh(road):
    """Extract speed from road feature, with fallback to default."""
    speed = road.get(SPEED_FIELD, DEFAULT_SPEED_KMH)
    if speed is None or (isinstance(speed, float) and np.isnan(speed)):
        return DEFAULT_SPEED_KMH
    try:
        speed = float(speed)
        if speed <= 0:
            return DEFAULT_SPEED_KMH
        return speed
    except (ValueError, TypeError):
        return DEFAULT_SPEED_KMH


def _flow_direction_to_links(road, start_coord, end_coord, nodes, next_link_id,
                              orig_start=None, orig_end=None):
    """
    Create directed link(s) based on FlowDirect attribute.
    Two-Way creates both directions; FromTo/ToFrom create one direction.
    
    Each link includes:
    - length: distance in meters
    - speed_kmh: speed limit from SpeedZone field
    - travel_time_min: length / speed (in minutes)
    
    Parameters:
        road: road feature with geometry and attributes
        start_coord: (x, y) of the start node (after clustering/remapping)
        end_coord: (x, y) of the end node (after clustering/remapping)
        nodes: dict mapping (x, y) -> node_id
        next_link_id: starting link ID
        orig_start: original (x, y) before remapping (to detect if adjustment needed)
        orig_end: original (x, y) before remapping (to detect if adjustment needed)
    
    Returns:
        list of link dicts, next_link_id
    """
    flow = str(road.get('FlowDirect', 'Two-Way')).strip()
    
    # Get geometry and adjust endpoints if they were remapped due to clustering
    geom = road.geometry
    start_changed = orig_start is not None and orig_start != start_coord
    end_changed = orig_end is not None and orig_end != end_coord
    
    if start_changed or end_changed:
        geom = _adjust_geometry_endpoints(geom, start_coord, end_coord)
    
    length_m = geom.length
    length_km = length_m / 1000.0
    speed_kmh = _get_speed_kmh(road)
    travel_time_hours = length_km / speed_kmh
    travel_time_min = travel_time_hours * 60
    
    link_type = road['CartoClass'].lower()
    name = road.get('StreetName', '')
    lanes = road.get('NumberofLa', 1)
    
    def one_link(from_coord, to_coord):
        return {
            'link_id': next_link_id,
            'from_node_id': nodes[from_coord],
            'to_node_id': nodes[to_coord],
            'directed': 1,
            'length': length_m,
            'speed_kmh': speed_kmh,
            'travel_time_min': round(travel_time_min, 4),
            'link_type': link_type,
            'name': name,
            'geometry': geom,
            'allowed_uses': 'car',
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
        # Unknown flow direction - treat as two-way
        links.append(one_link(start_coord, end_coord))
        next_link_id += 1
        links.append(one_link(end_coord, start_coord))
        next_link_id += 1
    
    return links, next_link_id


def create_road_edges(roads_gdf, nodes, next_link_id, coord_remap=None):
    """
    Create directed road edges from road geometries.
    
    Parameters:
        roads_gdf: GeoDataFrame of filtered roads
        nodes: dict mapping (x, y) -> node_id
        next_link_id: starting link ID
        coord_remap: optional dict mapping old (x, y) -> new (x, y) for clustered nodes
    
    Each edge includes:
    - length (m)
    - speed_kmh
    - travel_time_min
    
    Returns:
        road_edges_gdf, next_link_id
    """
    if coord_remap is None:
        coord_remap = {}
    
    road_edges = []
    
    for idx, road in roads_gdf.iterrows():
        coords = list(road.geometry.coords)
        orig_start = snap(coords[0])
        orig_end = snap(coords[-1])
        
        # Remap coordinates if nodes were clustered
        start_coord = coord_remap.get(orig_start, orig_start)
        end_coord = coord_remap.get(orig_end, orig_end)
        
        # Skip if either endpoint is not in the nodes dict
        if start_coord not in nodes or end_coord not in nodes:
            continue
        
        # Skip self-loops (can happen after clustering)
        if start_coord == end_coord:
            continue
        
        new_links, next_link_id = _flow_direction_to_links(
            road, start_coord, end_coord, nodes, next_link_id,
            orig_start=orig_start, orig_end=orig_end
        )
        road_edges.extend(new_links)
    
    road_edges_gdf = gpd.GeoDataFrame(road_edges, geometry='geometry', crs=roads_gdf.crs)
    return road_edges_gdf, next_link_id
