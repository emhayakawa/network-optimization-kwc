"""
Configuration for the bus-only network (GTFS-based).
"""
import os

# Transfer time when changing buses at the same stop (minutes)
TRANSFER_TIME_MINUTES = 15.0

# Walking links between nearby stops (improves connectivity)
WALKING_LINK_MAX_M = 200.0  # max straight-line distance (m) to add a walk link
WALKING_SPEED_M_PER_MIN = 80.0  # ~4.8 km/h; travel_time_min = distance_m / 80

# Stop clustering: nearby stops within this radius are grouped together
STOP_CLUSTER_RADIUS_M = 50.0  # meters

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
VALUE_OF_TIME = 0.50          # dollars per minute (= $30/hour)

# Note: Road network uses VALUE_OF_TIME = $25/hour = $0.417/min
# Transit typically uses same or slightly lower value

# Default path to GTFS zip (relative to URA folder)
_URA_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_GTFS_ZIP = os.path.join(_URA_DIR, "GRT_GTFS.zip")
