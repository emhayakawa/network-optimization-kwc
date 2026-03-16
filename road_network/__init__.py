"""
Road Network Package

Build and analyze road networks with:
- Road intersections from shapefiles
- Traffic lights as signalized nodes
- Node clustering for divided roads (dual carriageways)
- Generalized cost function for shortest path

Generalized Cost = (distance_km * COST_PER_KM) + (travel_time_hours * VALUE_OF_TIME)

Usage:
    from road_network import build_network, compute_shortest_path_networkx, compute_path_details
    
    project, nodes, edges = build_network()
    
    # Shortest path by generalized cost (using NetworkX - supports custom costs)
    result = compute_shortest_path_networkx(edges, orig, dest, weight_field='generalized_cost')
    
    # Or by distance
    result = compute_shortest_path_networkx(edges, orig, dest, weight_field='length')
    
    # Get cost breakdown
    details = compute_path_details(edges, result["path_links"])
    
    # AequilibraE version (only supports built-in fields like 'length')
    result = compute_shortest_path(project, orig, dest, cost_field="length")
"""

from .config import (
    ROADS_SHAPEFILE,
    TRAFFIC_LIGHTS_SHAPEFILE,
    GMNS_DIR,
    AEQUILIBRAE_PROJECT_DIR,
    ARCGIS_EXPORT_DIR,
    ALLOWED_ROAD_CLASSES,
    SNAP_TOLERANCE_M,
    NODE_CLUSTER_TOLERANCE_M,
    SIGNAL_DELAY_SECONDS,
    SPEED_FIELD,
    DEFAULT_SPEED_KMH,
    COST_PER_KM,
    VALUE_OF_TIME,
    PROJECT_CRS,
)

from .build_road_network import (
    build_network,
    load_and_clean_roads,
    save_gmns,
    create_aequilibrae_project,
    export_to_arcgis,
)

from .shortest_path import (
    compute_shortest_path,
    compute_shortest_path_networkx,
    build_networkx_graph,
    compute_path_details,
    compute_generalized_cost,
    add_generalized_cost_to_edges,
    export_path_to_gis,
)

from .nodes import (
    create_nodes_from_roads,
    load_traffic_lights,
    add_traffic_lights_to_nodes,
    cluster_nearby_nodes,
    merge_clustered_nodes,
    filter_to_main_component,
)

from .links import create_road_edges

__all__ = [
    # Main functions
    "build_network",
    "compute_shortest_path_networkx",  # Recommended - supports custom costs
    "compute_shortest_path",            # AequilibraE - only built-in fields
    "compute_path_details",
    "build_networkx_graph",
    "export_path_to_gis",
    # Config - Cost function
    "COST_PER_KM",
    "VALUE_OF_TIME",
    "SIGNAL_DELAY_SECONDS",
    # Config - Clustering
    "NODE_CLUSTER_TOLERANCE_M",
    # Config - Data
    "ROADS_SHAPEFILE",
    "TRAFFIC_LIGHTS_SHAPEFILE",
    "ALLOWED_ROAD_CLASSES",
    "SPEED_FIELD",
    "DEFAULT_SPEED_KMH",
    # Building blocks
    "load_and_clean_roads",
    "load_traffic_lights",
    "create_nodes_from_roads",
    "add_traffic_lights_to_nodes",
    "cluster_nearby_nodes",
    "merge_clustered_nodes",
    "create_road_edges",
    "compute_generalized_cost",
    "add_generalized_cost_to_edges",
    "save_gmns",
    "create_aequilibrae_project",
    "export_to_arcgis",
]
