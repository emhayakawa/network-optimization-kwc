"""
Configuration for the ION multimodal network (Bus + Light Rail Transit).

This module builds a combined transit network with:
- GRT Bus routes from Raw_GTFS
- ION LRT (Light Rail Transit) from GTFS(onlyLRT)
- ION extension stops from ION_Stops.csv
- Route geometry from ION_Routes.csv
"""
import os

# =============================================================================
# Data Paths
# =============================================================================
_URA_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# LRT GTFS data (ION Light Rail - current operating: Conestoga to Fairway)
LRT_GTFS_DIR = os.path.join(_URA_DIR, "GTFS(onlyLRT)")

# Bus GTFS data (GRT bus routes - full transit system)
BUS_GTFS_DIR = os.path.join(_URA_DIR, "Raw_GTFS")

# ION data files (stops and route geometry)
ION_STOPS_CSV = os.path.join(_URA_DIR, "ION_Stops.csv")
ION_ROUTES_CSV = os.path.join(_URA_DIR, "ION_Routes.csv")

# =============================================================================
# Network Parameters
# =============================================================================

# Transfer time between same-mode routes (minutes)
TRANSFER_TIME_BUS = 15.0      # Bus-to-bus transfer (same as bus_network)
TRANSFER_TIME_LRT = 5.0       # LRT-to-LRT transfer (faster)

# Transfer time between different modes (bus <-> LRT)
TRANSFER_TIME_BUS_LRT = 8.0   # Bus-to-LRT or LRT-to-bus

# Walking links between nearby stops (improves connectivity)
WALKING_LINK_MAX_M = 200.0    # max distance (m) to add a walk link
WALKING_SPEED_M_PER_MIN = 80.0  # ~4.8 km/h

# Stop clustering: nearby stops within this radius are grouped together
STOP_CLUSTER_RADIUS_M = 100.0  # meters (larger for multimodal transfers)

# =============================================================================
# Generalized Cost Parameters for Transit
# =============================================================================
# Generalized Cost = FARE + (WAITING_TIME + In-Vehicle Time) × VALUE_OF_TIME
#
# Formula breakdown:
#   - FARE: flat transit fare (one-time cost per trip)
#   - WAITING_TIME: average time waiting at origin (applied once)
#   - In-Vehicle Time: travel time on bus/LRT (from GTFS schedule)
#   - VALUE_OF_TIME: converts time to monetary units ($/min)

# Fare constants (GRT unified fare system)
FARE_CONSTANT = 3.50      # Flat fare in dollars (bus or LRT)
TRANSFER_FARE = 0.0       # Free transfers within GRT system

# Waiting time at origin stop (minutes)
# Bus: ~7.5 min average wait (15 min headway)
# LRT: ~5 min average wait (7-8 min headway during peak)
WAITING_TIME_BUS = 7.5    # Average wait for bus (minutes)
WAITING_TIME_LRT = 5.0    # Average wait for LRT (minutes)
WAITING_TIME_DEFAULT = 7.5  # Default for mixed trips

# Value of time (dollars per minute) - same as bus network
VALUE_OF_TIME = 0.33      # $20/hour

# =============================================================================
# Mode Identifiers
# =============================================================================

# Route ID prefix for ION LRT (to distinguish from bus routes)
LRT_ROUTE_PREFIX = "ION_"

# Mode types
MODE_BUS = "bus"
MODE_LRT = "lrt"

# GTFS route_type codes
GTFS_ROUTE_TYPE_LRT = 0   # Tram, Streetcar, Light rail
GTFS_ROUTE_TYPE_BUS = 3   # Bus

# =============================================================================
# Coordinate Reference Systems
# =============================================================================

# ION_Stops.csv uses UTM Zone 17N
ION_STOPS_CRS = "EPSG:26917"

# Output CRS for network exports
PROJECT_CRS = "EPSG:26917"

# WGS84 for GTFS data
WGS84_CRS = "EPSG:4326"
