"""
Preprocessing utilities for LRT (ION Light Rail) data.

This module:
1. Loads GTFS LRT stops (32 current stops from Conestoga to Fairway)
2. Loads ION_Stops.csv and filters only the 7 LRT extension stops (Cambridge extension)
3. Calculates average travel time from GTFS (used for extension links)
4. Creates extension links connecting new stops to the network
"""
import os
import sys
import pandas as pd
import numpy as np

# Handle both module import and direct execution
if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from config import (
        LRT_GTFS_DIR,
        ION_STOPS_CSV,
        ION_STOPS_CRS,
    )
else:
    # When imported from another module in same package, need to handle both cases
    try:
        from .config import (
            LRT_GTFS_DIR,
            ION_STOPS_CSV,
            ION_STOPS_CRS,
        )
    except ImportError:
        # Direct execution from parent importing this
        from config import (
            LRT_GTFS_DIR,
            ION_STOPS_CSV,
            ION_STOPS_CRS,
        )


def _parse_time(s):
    """Parse GTFS time 'H:MM:SS' or 'HH:MM:SS' to minutes since midnight."""
    if pd.isna(s) or s == "":
        return np.nan
    parts = str(s).strip().split(":")
    if len(parts) < 3:
        return np.nan
    h, m, sec = int(parts[0]), int(parts[1]), int(parts[2])
    return h * 60 + m + sec / 60.0


def _haversine_m(lat1, lon1, lat2, lon2):
    """Approximate distance in meters between two WGS84 points."""
    R = 6371000  # Earth radius in m
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c


def _convert_utm_to_wgs84(df):
    """Convert UTM coordinates (x_utm, y_utm) to WGS84 (stop_lat, stop_lon)."""
    try:
        from pyproj import Transformer
        
        # UTM Zone 17N to WGS84
        transformer = Transformer.from_crs(ION_STOPS_CRS, "EPSG:4326", always_xy=True)
        
        lons, lats = transformer.transform(df["x_utm"].values, df["y_utm"].values)
        df["stop_lon"] = lons
        df["stop_lat"] = lats
        
    except ImportError:
        print("Warning: pyproj not installed. Using approximate conversion.")
        # Approximate conversion for Waterloo region (UTM Zone 17N)
        df["stop_lon"] = (df["x_utm"] - 500000) / 111320 / np.cos(np.radians(43.5)) - 80.5
        df["stop_lat"] = df["y_utm"] / 111320
    
    return df


def load_ion_stops_csv(csv_path=None):
    """
    Load ION stops from the CSV file (ION_Stops.csv).
    
    Returns:
        DataFrame with columns: stop_name, x_utm, y_utm, stop_lat, stop_lon, etc.
    """
    path = csv_path or ION_STOPS_CSV
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"ION stops file not found: {path}")
    
    df = pd.read_csv(path)
    
    # Rename for consistency
    df = df.rename(columns={
        "X": "x_utm",
        "Y": "y_utm",
        "StopName": "stop_name",
        "Municipality": "municipality",
        "StopLocation": "stop_location",
        "StopStatus": "stop_status",
        "StopDirection": "stop_direction",
    })
    
    # Convert UTM to WGS84 (lat/lon)
    df = _convert_utm_to_wgs84(df)
    
    return df


def load_lrt_gtfs_stops(gtfs_dir=None):
    """
    Load stops from the LRT GTFS stops.txt file.
    
    Returns:
        DataFrame with GTFS stop columns (stop_id, stop_name, stop_lat, stop_lon, etc.)
    """
    path = gtfs_dir or LRT_GTFS_DIR
    stops_file = os.path.join(path, "stops.txt")
    
    if not os.path.exists(stops_file):
        raise FileNotFoundError(f"GTFS stops file not found: {stops_file}")
    
    df = pd.read_csv(stops_file)
    df["stop_id"] = df["stop_id"].astype(str)
    
    return df


def load_lrt_gtfs_stop_times(gtfs_dir=None):
    """Load stop_times.txt from LRT GTFS."""
    path = gtfs_dir or LRT_GTFS_DIR
    stop_times_file = os.path.join(path, "stop_times.txt")
    
    if not os.path.exists(stop_times_file):
        raise FileNotFoundError(f"GTFS stop_times file not found: {stop_times_file}")
    
    df = pd.read_csv(stop_times_file)
    df["stop_id"] = df["stop_id"].astype(str)
    
    return df


def calculate_avg_travel_time(gtfs_dir=None):
    """
    Calculate average travel time between consecutive LRT stops from GTFS.
    
    Returns:
        float: Average travel time in minutes
    """
    stop_times = load_lrt_gtfs_stop_times(gtfs_dir)
    stop_times["arrival_min"] = stop_times["arrival_time"].apply(_parse_time)
    stop_times["departure_min"] = stop_times["departure_time"].apply(_parse_time)
    stop_times = stop_times.sort_values(["trip_id", "stop_sequence"])
    
    travel_times = []
    
    for trip_id, trip_df in stop_times.groupby("trip_id"):
        trip_df = trip_df.sort_values("stop_sequence")
        arrivals = trip_df["arrival_min"].values
        departures = trip_df["departure_min"].values
        
        for i in range(len(arrivals) - 1):
            travel = arrivals[i + 1] - departures[i]
            if not np.isnan(travel) and travel > 0:
                travel_times.append(travel)
    
    avg_time = np.mean(travel_times) if travel_times else 2.5
    return avg_time


def get_lrt_extension_stops(ion_csv_path=None):
    """
    Get only the 7 LRT extension stops from ION_Stops.csv.
    
    These are the planned Cambridge extension stops (containing "(LRT)" in name):
    - Sportsworld (LRT)
    - Preston (LRT)
    - Eagle (LRT)
    - Delta (LRT)
    - Can-Amera (LRT)
    - Main (LRT)
    - Cambridge Terminus (LRT)
    
    Returns:
        DataFrame with extension stops
    """
    ion_stops = load_ion_stops_csv(ion_csv_path)
    
    # Filter only LRT extension stops (those with "(LRT)" in name)
    lrt_ext = ion_stops[ion_stops["stop_name"].str.contains(r"\(LRT\)", regex=True, case=False)].copy()
    
    # Remove "(LRT)" suffix for cleaner names
    lrt_ext["stop_name"] = lrt_ext["stop_name"].str.replace(r"\s*\(LRT\)", "", regex=True).str.strip()
    
    return lrt_ext


def preprocess_lrt_stops(gtfs_dir=None, ion_csv_path=None):
    """
    Preprocess LRT stops by loading both GTFS and extension stops.
    
    Returns:
        dict with gtfs_stops, extension_stops, avg_travel_time_min
    """
    gtfs_stops = load_lrt_gtfs_stops(gtfs_dir)
    extension_stops = get_lrt_extension_stops(ion_csv_path)
    avg_time = calculate_avg_travel_time(gtfs_dir)
    
    print(f"  GTFS LRT stops: {len(gtfs_stops)}")
    print(f"  Extension stops: {len(extension_stops)}")
    print(f"  Average travel time: {avg_time:.2f} min")
    
    return {
        "gtfs_stops": gtfs_stops,
        "extension_stops": extension_stops,
        "avg_travel_time_min": avg_time,
    }


def merge_lrt_stops_with_gtfs(gtfs_dir=None, ion_csv_path=None, output_dir=None):
    """
    Merge GTFS LRT stops with the 7 extension stops.
    Creates augmented stops and extension links.
    
    Parameters:
        gtfs_dir: Path to LRT GTFS directory
        ion_csv_path: Path to ION_Stops.csv
        output_dir: Where to save output files (default: ion_network/data)
    
    Returns:
        dict with:
        - augmented_stops: DataFrame with all stops (GTFS + extension)
        - extension_links: DataFrame with links for the extension
    """
    result = preprocess_lrt_stops(gtfs_dir, ion_csv_path)
    
    gtfs_stops = result["gtfs_stops"]
    extension_stops = result["extension_stops"]
    avg_time = result["avg_travel_time_min"]
    
    # Assign stop IDs to extension stops (start from 10001 to avoid conflicts)
    ext_start_id = 10001
    extension_stops = extension_stops.reset_index(drop=True)
    extension_stops["stop_id"] = [str(ext_start_id + i) for i in range(len(extension_stops))]
    
    # Order extension stops from north to south (Sportsworld closest to Fairway)
    extension_order = [
        "Sportsworld",
        "Preston",
        "Pinebush",
        "Cambridge Centre Mall",
        "Can-Amera",
        "Delta",
        "Cambridge Terminus"
    ]
    
    # Create ordered DataFrame
    ordered_ext = []
    for name in extension_order:
        match = extension_stops[extension_stops["stop_name"].str.contains(name, case=False)]
        if len(match) > 0:
            ordered_ext.append(match.iloc[0])
    
    extension_stops = pd.DataFrame(ordered_ext)
    extension_stops = extension_stops.reset_index(drop=True)
    extension_stops["stop_id"] = [str(ext_start_id + i) for i in range(len(extension_stops))]
    
    print(f"\n  Extension stops (ordered):")
    for _, row in extension_stops.iterrows():
        print(f"    {row['stop_id']}: {row['stop_name']}")
    
    # Create augmented stops DataFrame
    gtfs_cols = ["stop_id", "stop_name", "stop_lat", "stop_lon"]
    ext_cols = ["stop_id", "stop_name", "stop_lat", "stop_lon"]
    
    augmented = pd.concat([
        gtfs_stops[gtfs_cols],
        extension_stops[ext_cols]
    ], ignore_index=True)
    
    print(f"\n  Total augmented stops: {len(augmented)}")
    
    # Create extension links
    # 1. Link first extension stop (Sportsworld) to Fairway Station (both directions)
    # 2. Link extension stops sequentially
    
    extension_links = []
    
    # Find Fairway Station stops in GTFS (there are 2: Northbound and Southbound)
    fairway_stops = gtfs_stops[gtfs_stops["stop_name"].str.contains("Fairway", case=False)]
    first_ext_id = extension_stops.iloc[0]["stop_id"]
    first_ext_lat = float(extension_stops.iloc[0]["stop_lat"])
    first_ext_lon = float(extension_stops.iloc[0]["stop_lon"])
    
    print(f"\n  Connecting to Fairway Station:")
    for _, fw_row in fairway_stops.iterrows():
        fw_id = fw_row["stop_id"]
        fw_lat = float(fw_row["stop_lat"])
        fw_lon = float(fw_row["stop_lon"])
        dist_m = _haversine_m(fw_lat, fw_lon, first_ext_lat, first_ext_lon)
        
        print(f"    {fw_row['stop_name']} ({fw_id}) <-> Sportsworld ({first_ext_id}): {dist_m:.0f} m")
        
        # Bidirectional links
        extension_links.append({
            "from_stop_id": str(fw_id),
            "to_stop_id": str(first_ext_id),
            "travel_time_min": round(avg_time, 2),
            "length_m": round(dist_m, 2),
            "link_type": "route",
            "mode": "lrt",
        })
        extension_links.append({
            "from_stop_id": str(first_ext_id),
            "to_stop_id": str(fw_id),
            "travel_time_min": round(avg_time, 2),
            "length_m": round(dist_m, 2),
            "link_type": "route",
            "mode": "lrt",
        })
    
    # Sequential extension links
    print(f"\n  Extension segment links:")
    for i in range(len(extension_stops) - 1):
        from_row = extension_stops.iloc[i]
        to_row = extension_stops.iloc[i + 1]
        
        from_id = from_row["stop_id"]
        to_id = to_row["stop_id"]
        
        dist_m = _haversine_m(
            float(from_row["stop_lat"]), float(from_row["stop_lon"]),
            float(to_row["stop_lat"]), float(to_row["stop_lon"])
        )
        
        print(f"    {from_row['stop_name']} -> {to_row['stop_name']}: {dist_m:.0f} m, {avg_time:.2f} min")
        
        # Bidirectional
        extension_links.append({
            "from_stop_id": str(from_id),
            "to_stop_id": str(to_id),
            "travel_time_min": round(avg_time, 2),
            "length_m": round(dist_m, 2),
            "link_type": "route",
            "mode": "lrt",
        })
        extension_links.append({
            "from_stop_id": str(to_id),
            "to_stop_id": str(from_id),
            "travel_time_min": round(avg_time, 2),
            "length_m": round(dist_m, 2),
            "link_type": "route",
            "mode": "lrt",
        })
    
    extension_links_df = pd.DataFrame(extension_links)
    print(f"\n  Total extension links: {len(extension_links_df)}")
    
    # Save to output directory
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(output_dir, exist_ok=True)
    
    stops_path = os.path.join(output_dir, "stops_augmented.csv")
    links_path = os.path.join(output_dir, "extension_links.csv")
    
    augmented.to_csv(stops_path, index=False)
    extension_links_df.to_csv(links_path, index=False)
    
    print(f"\n  Saved: {stops_path}")
    print(f"  Saved: {links_path}")
    
    return {
        "augmented_stops": augmented,
        "extension_links": extension_links_df,
    }


if __name__ == "__main__":
    print("=== LRT Stop Preprocessing ===\n")
    
    result = merge_lrt_stops_with_gtfs()
    
    print("\n=== Summary ===")
    print(f"Augmented stops: {len(result['augmented_stops'])}")
    print(f"Extension links: {len(result['extension_links'])}")
