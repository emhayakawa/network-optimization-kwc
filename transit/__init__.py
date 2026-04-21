"""
Shared transit network utilities (bus and ION multimodal).

Zone-to-zone routing lives in ``zone_to_zone_routing`` (repo root); it is re-exported here
for convenience. Node-to-node APIs are unchanged.
"""
import os
import sys

_URA_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _URA_ROOT not in sys.path:
    sys.path.insert(0, _URA_ROOT)

from .shortest_path import (
    shortest_path_transit,
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

# zone_to_zone_routing imports transit.shortest_path; eager re-import here caused a circular
# import. Lazy attributes keep ``from transit import shortest_path_road_zone_to_zone`` working.
_Z2Z_EXPORTS = frozenset(
    {
        "node_ids_in_zone",
        "shortest_path_transit_zone_to_zone",
        "shortest_path_ion_zone_to_zone",
        "shortest_path_road_zone_to_zone",
        "shortest_path_zone_to_zone",
    }
)


def __getattr__(name):
    if name in _Z2Z_EXPORTS:
        import zone_to_zone_routing as _z2z

        return getattr(_z2z, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "shortest_path_transit",
    "shortest_path_transit_zone_to_zone",
    "shortest_path_ion_zone_to_zone",
    "shortest_path_road_zone_to_zone",
    "shortest_path_zone_to_zone",
    "compute_generalized_cost",
    "compute_path_details",
    "node_ids_at_stop",
    "node_ids_in_zone",
    "node_ids_by_mode",
    "check_connectivity",
    "export_shortest_path_to_arcgis",
    "COST_TIME",
    "COST_DISTANCE",
    "COST_GENERALIZED",
]
