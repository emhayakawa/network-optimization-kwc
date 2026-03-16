"""
Configuration for the road network (roads + traffic lights).
Adjust these constants to change network behavior.
"""
import os

# ============================================================================
# File Paths (relative to URA folder)
# ============================================================================
_URA_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA_DIR = os.path.join(_URA_DIR, "Data")

ROADS_SHAPEFILE = os.path.join(_DATA_DIR, "Roads/Roads.shp")
TRAFFIC_LIGHTS_SHAPEFILE = os.path.join(_DATA_DIR, "Traffic_Lights_-7232028556142916675/Traffic_Lights.shp")

# Output directories
GMNS_DIR = os.path.join(_URA_DIR, "road_network/data/gmns")
AEQUILIBRAE_PROJECT_DIR = os.path.join(_URA_DIR, "road_network/aequilibrae_project")
ARCGIS_EXPORT_DIR = os.path.join(_URA_DIR, "road_network/arcgis_export")

# ============================================================================
# Road Filtering
# ============================================================================
ALLOWED_ROAD_CLASSES = ['Freeway', 'Highway', 'Ramp', 'Arterial']
# Add 'Collector', 'Localstreet' if you want more road coverage

# ============================================================================
# Road Data Fields
# ============================================================================
SPEED_FIELD = "SpeedZone"  # Field name for speed limit in roads shapefile (km/h)
DEFAULT_SPEED_KMH = 50     # Default speed if SpeedZone is missing or invalid

# ============================================================================
# Node Clustering (for divided roads / dual carriageways)
# ============================================================================
# Nodes within this distance are merged into a single intersection node
# Set to 0 to disable clustering
NODE_CLUSTER_TOLERANCE_M = 30  # meters - handles divided road intersections

# ============================================================================
# Node Snapping
# ============================================================================
SNAP_TOLERANCE_M = 30  # traffic lights within this distance of intersection snap to it

# ============================================================================
# Traffic Signal Delay (for generalized cost)
# ============================================================================
SIGNAL_DELAY_SECONDS = 40  # average wait time at signalized intersection

# ============================================================================
# Generalized Cost Function Parameters
# ============================================================================
# Generalized Cost = (distance_km * COST_PER_KM) + (travel_time_hours * VALUE_OF_TIME)
COST_PER_KM = 0.50         # $/km - vehicle operating cost (fuel, wear, etc.)
VALUE_OF_TIME = 40.00      # $/hour - value of travel time

# ============================================================================
# Spatial Reference
# ============================================================================
PROJECT_CRS = "EPSG:26917"  # UTM Zone 17N (Ontario)
OUTPUT_CRS = "EPSG:4326"    # WGS84 for exports
