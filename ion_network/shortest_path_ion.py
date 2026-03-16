"""
ION network shortest path - re-exports from shared transit module.
Kept for backward compatibility (direct imports from shortest_path_ion).
"""
import os
import sys
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

from transit.shortest_path import (
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

# Backward compatibility
shortest_path_ion = shortest_path_transit
