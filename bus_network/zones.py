"""
TAZ (Traffic Analysis Zone) handling for the bus network.

Loads TAZ polygons and assigns a zone_id to points or nodes (e.g. bus network nodes at
cluster lat/lon) via spatial join. No zone centroids are required for routing; use
``zone_to_zone_routing`` (repo root) for zone-to-zone shortest paths on bus, ION, or road networks.
"""
import os

import geopandas as gpd
import pandas as pd

_URA_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA_DIR = os.path.join(_URA_DIR, "Data")
DEFAULT_TAZ_SHAPEFILE = os.path.join(
    _DATA_DIR, "2011 RMOW RTM TAZ_zone", "2011 RMOW RTM TAZ_zone.shp"
)

PROJECT_CRS = "EPSG:26917"

_ZONE_ID_CANDIDATES = [
    "ID_TAZ",
    "TAZ",
    "TAZ_ID",
    "TAZID",
    "Zone",
    "ZONE",
    "ZoneID",
    "ID",
]


def load_taz_zones(taz_path=None):
    """
    Load TAZ zones from a shapefile (or other file geopandas.read_file supports).

    Returns:
        GeoDataFrame in PROJECT_CRS (EPSG:26917).
    """
    path = taz_path or DEFAULT_TAZ_SHAPEFILE
    zones_gdf = gpd.read_file(path)

    if zones_gdf.crs is None:
        zones_gdf = zones_gdf.set_crs(PROJECT_CRS)
    elif zones_gdf.crs != PROJECT_CRS:
        zones_gdf = zones_gdf.to_crs(PROJECT_CRS)

    print(f"Loaded {len(zones_gdf)} TAZ zones from {path}")
    print(f"Columns: {zones_gdf.columns.tolist()}")

    return zones_gdf


def detect_zone_id_field(zones_gdf, zone_id_field=None):
    """Return the attribute column name used as zone identifier."""
    if zone_id_field is not None:
        if zone_id_field not in zones_gdf.columns:
            raise KeyError(f"zone_id_field {zone_id_field!r} not in zones columns")
        return zone_id_field
    for col in _ZONE_ID_CANDIDATES:
        if col in zones_gdf.columns:
            return col
    raise ValueError(
        "Could not detect zone ID column in TAZ layer. "
        f"Columns: {zones_gdf.columns.tolist()}. Pass zone_id_field=..."
    )


def assign_zone_id_by_location(
    df,
    zones_gdf,
    lat_col="stop_lat",
    lon_col="stop_lon",
    zone_id_field=None,
):
    """
    Add zone_id to each row from a point-in-polygon join (EPSG:4326 lon/lat → zones CRS).

    Points outside all polygons are assigned to the nearest zone polygon.

    Parameters:
        df: DataFrame with lat_col, lon_col in WGS84
        zones_gdf: TAZ polygons (typically from load_taz_zones)
        zone_id_field: attribute for zone id; auto-detected if None

    Returns:
        DataFrame copy with int zone_id column.
    """
    zone_id_field = detect_zone_id_field(zones_gdf, zone_id_field)
    out = df.copy()

    points = gpd.GeoDataFrame(
        out,
        geometry=gpd.points_from_xy(out[lon_col], out[lat_col]),
        crs="EPSG:4326",
    )
    points = points.to_crs(zones_gdf.crs)

    joined = gpd.sjoin(
        points,
        zones_gdf[[zone_id_field, "geometry"]],
        how="left",
        predicate="within",
    )
    joined = joined.rename(columns={zone_id_field: "zone_id"})

    # Overlapping TAZ polygons can match the same point more than once; keep one row per input.
    if joined.index.duplicated().any():
        n_before = len(joined)
        prefer = joined.assign(_z=joined["zone_id"].notna()).sort_values(
            "_z", ascending=False
        )
        joined = prefer[~prefer.index.duplicated(keep="first")].drop(columns=["_z"])
        if n_before > len(joined):
            print(
                f"  {n_before - len(joined)} duplicate TAZ matches dropped "
                f"(overlapping polygons; kept one zone per point)"
            )

    unassigned = joined["zone_id"].isna()
    if unassigned.any():
        print(
            f"  {unassigned.sum()} points outside TAZ boundaries — assigning nearest zone"
        )
        for idx in joined[unassigned].index:
            g = joined.loc[idx, "geometry"]
            distances = zones_gdf.geometry.distance(g)
            nearest_idx = distances.idxmin()
            joined.loc[idx, "zone_id"] = zones_gdf.loc[nearest_idx, zone_id_field]

    if "index_right" in joined.columns:
        joined = joined.drop(columns=["index_right"])

    joined = joined.drop(columns=["geometry"])
    joined["zone_id"] = joined["zone_id"].astype(int)
    print(
        f"Assigned {len(joined)} locations to {joined['zone_id'].nunique()} zones"
    )
    return joined


def assign_zone_id_to_point_geodataframe(gdf, zones_gdf, zone_id_field=None):
    """
    Add zone_id to a GeoDataFrame of points (any CRS). Uses WGS84 coordinates for the TAZ join.

    Typical use: road network nodes after build (geometry in EPSG:26917).
    """
    if gdf.crs is None:
        raise ValueError("GeoDataFrame must have a CRS set")
    wgs = gdf.to_crs("EPSG:4326")
    tmp = pd.DataFrame(index=gdf.index)
    tmp["stop_lon"] = wgs.geometry.x.values
    tmp["stop_lat"] = wgs.geometry.y.values
    assigned = assign_zone_id_by_location(tmp, zones_gdf, zone_id_field=zone_id_field)
    out = gdf.copy()
    out["zone_id"] = assigned.loc[gdf.index, "zone_id"].values
    return out


def assign_stops_to_zones(stops_gdf, zones_gdf, zone_id_field=None):
    """
    Assign each bus stop (GeoDataFrame with geometry) to a TAZ zone.

    Returns:
        GeoDataFrame with zone_id column.
    """
    zone_id_field = detect_zone_id_field(zones_gdf, zone_id_field)

    if stops_gdf.crs != zones_gdf.crs:
        stops_gdf = stops_gdf.to_crs(zones_gdf.crs)

    stops_with_zones = gpd.sjoin(
        stops_gdf,
        zones_gdf[[zone_id_field, "geometry"]],
        how="left",
        predicate="within",
    )
    stops_with_zones = stops_with_zones.rename(columns={zone_id_field: "zone_id"})

    if stops_with_zones.index.duplicated().any():
        prefer = stops_with_zones.assign(
            _z=stops_with_zones["zone_id"].notna()
        ).sort_values("_z", ascending=False)
        stops_with_zones = prefer[
            ~prefer.index.duplicated(keep="first")
        ].drop(columns=["_z"])

    unassigned = stops_with_zones["zone_id"].isna()
    if unassigned.any():
        print(
            f"  {unassigned.sum()} stops outside TAZ boundaries — assigning nearest zone"
        )
        for idx in stops_with_zones[unassigned].index:
            stop_geom = stops_with_zones.loc[idx, "geometry"]
            distances = zones_gdf.geometry.distance(stop_geom)
            nearest_idx = distances.idxmin()
            stops_with_zones.loc[idx, "zone_id"] = zones_gdf.loc[
                nearest_idx, zone_id_field
            ]

    if "index_right" in stops_with_zones.columns:
        stops_with_zones = stops_with_zones.drop(columns=["index_right"])

    stops_with_zones["zone_id"] = stops_with_zones["zone_id"].astype(int)
    print(
        f"Assigned {len(stops_with_zones)} stops to "
        f"{stops_with_zones['zone_id'].nunique()} zones"
    )
    return stops_with_zones


def get_zone_stop_counts(stops_with_zones):
    """DataFrame: zone_id, stop_count."""
    return stops_with_zones.groupby("zone_id").size().reset_index(name="stop_count")


def export_zones_to_gis(zones_gdf, output_path, centroids_gdf=None):
    """
    Export **TAZ polygon boundaries** to a GeoPackage (layer ``zones``, WGS84), not transit/road nodes.

    The ``zone_id`` attribute on network nodes comes from ``build_bus_network`` / ``build_ion_network`` /
    ``build_road_network``, which write the ``nodes`` layer in each network's ``arcgis_export/*.gpkg``
    (from ``node.csv`` / node GeoDataFrame). Use this helper only if you want a standalone TAZ map.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    if os.path.exists(output_path):
        os.remove(output_path)

    zones_wgs = zones_gdf.to_crs("EPSG:4326")
    zones_wgs.to_file(output_path, layer="zones", driver="GPKG")

    if centroids_gdf is not None and len(centroids_gdf):
        centroids_wgs = centroids_gdf.to_crs("EPSG:4326")
        centroids_wgs.to_file(output_path, layer="centroids", driver="GPKG", mode="a")
        print(f"  - Layer 'centroids': {len(centroids_gdf)} points")

    print(f"Exported zones to: {output_path}")
    print(f"  - Layer 'zones': {len(zones_gdf)} TAZ polygons")


if __name__ == "__main__":
    print("=== Loading TAZ Zones ===")
    z = load_taz_zones()
    export_path = os.path.join(_URA_DIR, "bus_network/arcgis_export/taz_zones.gpkg")
    print("\n=== Exporting polygons only (no centroids) ===")
    export_zones_to_gis(z, export_path)
