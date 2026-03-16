"""
TAZ (Traffic Analysis Zone) handling for the bus network.

Creates zone centroids as nodes and maps bus stops to their containing zones.
This simplifies the network by using zone-to-zone routing instead of stop-to-stop.
"""
import os
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

# Default path to TAZ shapefile (relative to URA folder)
_URA_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_TAZ_SHAPEFILE = os.path.join(_URA_DIR, "2011 RMOW RTM TAZ_zone/2011 RMOW RTM TAZ_zone.shp")

# CRS for the project
PROJECT_CRS = "EPSG:26917"


def load_taz_zones(taz_path=None):
    """
    Load TAZ zones from shapefile.
    
    Parameters:
        taz_path: path to TAZ shapefile (default: 2011 RMOW RTM TAZ_zone)
    
    Returns:
        GeoDataFrame of TAZ zones with geometry
    """
    path = taz_path or DEFAULT_TAZ_SHAPEFILE
    zones_gdf = gpd.read_file(path)
    
    # Ensure correct CRS
    if zones_gdf.crs is None:
        zones_gdf = zones_gdf.set_crs(PROJECT_CRS)
    elif zones_gdf.crs != PROJECT_CRS:
        zones_gdf = zones_gdf.to_crs(PROJECT_CRS)
    
    print(f"Loaded {len(zones_gdf)} TAZ zones")
    print(f"Columns: {zones_gdf.columns.tolist()}")
    
    return zones_gdf


def create_zone_centroids(zones_gdf, zone_id_field=None):
    """
    Create centroid points for each TAZ zone.
    
    Parameters:
        zones_gdf: GeoDataFrame of TAZ zones
        zone_id_field: column name for zone ID (auto-detected if None)
    
    Returns:
        GeoDataFrame of centroids with zone_id, geometry (Point)
    """
    # Auto-detect zone ID field
    if zone_id_field is None:
        # Look for common zone ID field names
        candidates = ['TAZ', 'TAZ_ID', 'TAZID', 'Zone', 'ZONE', 'ZoneID', 'ID', 'FID']
        for col in candidates:
            if col in zones_gdf.columns:
                zone_id_field = col
                break
        if zone_id_field is None:
            # Use first numeric column or index
            numeric_cols = zones_gdf.select_dtypes(include=['int64', 'int32', 'float64']).columns
            if len(numeric_cols) > 0:
                zone_id_field = numeric_cols[0]
            else:
                zones_gdf = zones_gdf.reset_index()
                zone_id_field = 'index'
    
    print(f"Using '{zone_id_field}' as zone ID field")
    
    # Compute centroids
    centroids = zones_gdf.copy()
    centroids['centroid_geom'] = centroids.geometry.centroid
    
    # Create centroid GeoDataFrame
    centroids_gdf = gpd.GeoDataFrame(
        {
            'zone_id': centroids[zone_id_field].astype(int),
            'x_coord': centroids['centroid_geom'].x,
            'y_coord': centroids['centroid_geom'].y,
            'geometry': centroids['centroid_geom'],
        },
        crs=zones_gdf.crs
    )
    
    # Add original zone area for reference
    centroids_gdf['zone_area_sqm'] = zones_gdf.geometry.area
    
    print(f"Created {len(centroids_gdf)} zone centroids")
    
    return centroids_gdf


def assign_stops_to_zones(stops_gdf, zones_gdf, zone_id_field=None):
    """
    Assign each bus stop to its containing TAZ zone.
    
    Parameters:
        stops_gdf: GeoDataFrame of bus stops (must have geometry)
        zones_gdf: GeoDataFrame of TAZ zones
        zone_id_field: column name for zone ID in zones_gdf
    
    Returns:
        stops_gdf with added 'zone_id' column
    """
    # Auto-detect zone ID field if not provided
    if zone_id_field is None:
        candidates = ['TAZ', 'TAZ_ID', 'TAZID', 'Zone', 'ZONE', 'ZoneID', 'ID']
        for col in candidates:
            if col in zones_gdf.columns:
                zone_id_field = col
                break
    
    # Ensure same CRS
    if stops_gdf.crs != zones_gdf.crs:
        stops_gdf = stops_gdf.to_crs(zones_gdf.crs)
    
    # Spatial join to find containing zone
    stops_with_zones = gpd.sjoin(
        stops_gdf,
        zones_gdf[[zone_id_field, 'geometry']],
        how='left',
        predicate='within'
    )
    
    # Rename zone ID column
    stops_with_zones = stops_with_zones.rename(columns={zone_id_field: 'zone_id'})
    
    # Handle stops outside all zones (assign to nearest zone)
    unassigned = stops_with_zones['zone_id'].isna()
    if unassigned.any():
        print(f"  {unassigned.sum()} stops outside TAZ boundaries - assigning to nearest zone")
        for idx in stops_with_zones[unassigned].index:
            stop_geom = stops_with_zones.loc[idx, 'geometry']
            distances = zones_gdf.geometry.distance(stop_geom)
            nearest_idx = distances.idxmin()
            stops_with_zones.loc[idx, 'zone_id'] = zones_gdf.loc[nearest_idx, zone_id_field]
    
    # Clean up sjoin columns
    if 'index_right' in stops_with_zones.columns:
        stops_with_zones = stops_with_zones.drop(columns=['index_right'])
    
    stops_with_zones['zone_id'] = stops_with_zones['zone_id'].astype(int)
    
    assigned_zones = stops_with_zones['zone_id'].nunique()
    print(f"Assigned {len(stops_with_zones)} stops to {assigned_zones} zones")
    
    return stops_with_zones


def get_zone_stop_counts(stops_with_zones):
    """
    Get count of stops per zone.
    
    Returns:
        DataFrame with zone_id and stop_count
    """
    counts = stops_with_zones.groupby('zone_id').size().reset_index(name='stop_count')
    return counts


def export_zones_to_gis(zones_gdf, centroids_gdf, output_path):
    """
    Export zones and centroids to GeoPackage for visualization.
    
    Parameters:
        zones_gdf: GeoDataFrame of TAZ polygons
        centroids_gdf: GeoDataFrame of zone centroids
        output_path: path to output .gpkg file
    """
    import os
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    
    # Remove existing file
    if os.path.exists(output_path):
        os.remove(output_path)
    
    # Export to WGS84 for ArcGIS
    zones_wgs = zones_gdf.to_crs("EPSG:4326")
    centroids_wgs = centroids_gdf.to_crs("EPSG:4326")
    
    zones_wgs.to_file(output_path, layer="zones", driver="GPKG")
    centroids_wgs.to_file(output_path, layer="centroids", driver="GPKG", mode="a")
    
    print(f"Exported zones to: {output_path}")
    print(f"  - Layer 'zones': {len(zones_gdf)} TAZ polygons")
    print(f"  - Layer 'centroids': {len(centroids_gdf)} zone centroids")


if __name__ == "__main__":
    # Test zone loading and centroid creation
    print("=== Loading TAZ Zones ===")
    zones_gdf = load_taz_zones()
    
    print("\n=== Creating Zone Centroids ===")
    centroids_gdf = create_zone_centroids(zones_gdf)
    
    print("\n=== Zone Centroid Sample ===")
    print(centroids_gdf.head(10))
    
    print("\n=== Exporting to GeoPackage ===")
    export_path = os.path.join(_URA_DIR, "bus_network/arcgis_export/taz_zones.gpkg")
    export_zones_to_gis(zones_gdf, centroids_gdf, export_path)
