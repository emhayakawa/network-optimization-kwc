"""
ION Network: Multimodal Transit Network (Bus + Light Rail Transit)

This module builds a multimodal transit network combining:
- GRT Bus routes from Raw_GTFS folder
- ION LRT (Light Rail Transit) from GTFS(onlyLRT) folder
- ION extension stops from ION_Stops.csv
- Route geometry from ION_Routes.csv

The network supports:
- Shortest path computation using generalized cost (same as bus network)
- Multimodal transfers between bus and LRT
- Export to ArcGIS/QGIS (GeoPackage format)
- AequilibraE project creation for traffic assignment (merge with road network)
"""

from .build_ion_network import (
    load_bus_gtfs,
    load_lrt_gtfs,
    load_ion_routes,
    load_ion_stops,
    build_bus_network_component,
    build_lrt_network_component,
    merge_networks,
    build_nodes_and_links,
    save_network,
    export_to_arcgis,
    export_to_gmns,
    build_network,
)
from .aequilibrae_network import create_aequilibrae_project
from .shortest_path_ion import (
    shortest_path_ion,
    compute_generalized_cost,
    compute_path_details,
    node_ids_at_stop,
    node_ids_by_mode,
    check_connectivity,
    export_shortest_path_to_arcgis,
    COST_TIME,
    COST_DISTANCE,
    COST_GENERALIZED,
)
from .preprocess_lrt import preprocess_lrt_stops, merge_lrt_stops_with_gtfs

__all__ = [
    # Data loading
    "load_bus_gtfs",
    "load_lrt_gtfs",
    "load_ion_routes",
    "load_ion_stops",
    # Network building
    "build_bus_network_component",
    "build_lrt_network_component",
    "merge_networks",
    "build_nodes_and_links",
    "save_network",
    "export_to_arcgis",
    "export_to_gmns",
    "build_network",
    "create_aequilibrae_project",
    # Shortest path
    "shortest_path_ion",
    "compute_generalized_cost",
    "compute_path_details",
    "node_ids_at_stop",
    "node_ids_by_mode",
    "check_connectivity",
    "export_shortest_path_to_arcgis",
    # Cost types
    "COST_TIME",
    "COST_DISTANCE",
    "COST_GENERALIZED",
    # Preprocessing
    "preprocess_lrt_stops",
    "merge_lrt_stops_with_gtfs",
]
