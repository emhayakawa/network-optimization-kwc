"""
Configuration for the bus-only network (GTFS-based).
"""
import os

# Transfer time when changing buses at the same stop (minutes)
TRANSFER_TIME_MINUTES = 15.0

# Stop clustering: nearby stops within this radius are grouped together
STOP_CLUSTER_RADIUS_M = 100.0  # meters

# =============================================================================
# Generalized Cost Parameters for Transit
# =============================================================================
# Generalized Cost = FARE + (WAITING_TIME + In-Vehicle Time) × VALUE_OF_TIME
#
# Formula breakdown:
#   - FARE: flat transit fare (one-time cost per trip)
#   - WAITING_TIME: average time waiting for bus at origin (applied once)
#   - In-Vehicle Time: travel time on the bus (from GTFS schedule)
#   - VALUE_OF_TIME: converts time to monetary units ($/min)

FARE_CONSTANT = 3.50          # flat fare in dollars
WAITING_TIME_MINUTES = 7.5    # average wait time for bus at origin (minutes)
VALUE_OF_TIME = 0.33          # dollars per minute ($20/hour)

# Note: Road network uses VALUE_OF_TIME = $25/hour = $0.417/min
# Transit typically uses same or slightly lower value

# Default path to GTFS data (directory or zip)
_URA_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA_DIR = os.path.join(_URA_DIR, "Data")
# Raw_GTFS directory (same as ion_network bus data)
DEFAULT_GTFS_PATH = os.path.join(_DATA_DIR, "Raw_GTFS")

# TAZ polygons for assigning zone_id to bus nodes (spatial join at node coordinates)
DEFAULT_TAZ_SHAPEFILE = os.path.join(
    _DATA_DIR, "2011 RMOW RTM TAZ_zone", "2011 RMOW RTM TAZ_zone.shp"
)
