"""
Edit these values, then from the URA repo root run:

    python zone_to_zone_routing.py

CLI arguments override this file, e.g. ``python zone_to_zone_routing.py --dest 21414``.
"""

# --- Which network to load: "bus" | "ion" | "road" ---
NETWORK = "bus"

# --- Zones (TAZ / zone_id on your node tables) ---
ORIG_ZONE_ID = 22001
DEST_ZONE_ID = 21228

# --- Transit (bus / ion): "generalized" | "time" | "distance" ---
TRANSIT_COST = "generalized"

# --- Road only: column on links for NetworkX shortest path ---
ROAD_WEIGHT_FIELD = "generalized_cost"

# --- Print every successful node-to-node OD in the zone pair (sorted; * = shortest) ---
LIST_ALL_CANDIDATE_PATHS = True

# --- Optional path overrides (None = default under this repo) ---
BUS_NODE_CSV = None
BUS_LINK_CSV = None
ION_NODE_CSV = None
ION_LINK_CSV = None
ROAD_GPKG_PATH = None
