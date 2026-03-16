"""
Bus-only network: nodes = (stop, route), links = route segments + transfer links (15 min).
Route segments are built in both directions (buses run each way).
Shortest path: use AequilibraE (shortest_path_bus_aequilibrae) for assignment compatibility.
"""
import os
import sys
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

from .config import TRANSFER_TIME_MINUTES, DEFAULT_GTFS_PATH
from .build_bus_network import load_gtfs, build_nodes_and_links, save_network, export_to_gmns, export_to_arcgis
from transit.shortest_path import (
    shortest_path_transit,
    compute_generalized_cost,
    compute_path_details,
    node_ids_at_stop,
    check_connectivity,
    export_shortest_path_to_arcgis,
    COST_TIME,
    COST_DISTANCE,
    COST_GENERALIZED,
)
from .shortest_path_bus import shortest_path_bus, shortest_path_bus_aequilibrae
from .aequilibrae_network import create_aequilibrae_project

__all__ = [
    "TRANSFER_TIME_MINUTES",
    "DEFAULT_GTFS_PATH",
    "load_gtfs",
    "build_nodes_and_links",
    "save_network",
    "export_to_gmns",
    "export_to_arcgis",
    "shortest_path_transit",
    "shortest_path_bus",
    "shortest_path_bus_aequilibrae",
    "compute_generalized_cost",
    "compute_path_details",
    "node_ids_at_stop",
    "check_connectivity",
    "export_shortest_path_to_arcgis",
    "COST_TIME",
    "COST_DISTANCE",
    "COST_GENERALIZED",
    "create_aequilibrae_project",
]
