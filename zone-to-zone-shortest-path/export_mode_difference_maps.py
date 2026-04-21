"""
Export map-ready zone-level generalized-cost difference layers (ArcGIS/QGIS).

Creates a GeoPackage with zone-level choropleths (origin-zone summaries of mode deltas).
Only zones that appear in the analyzed OD results for each mode comparison are exported.

Usage:
    python zone-to-zone-shortest-path/export_mode_difference_maps.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import pandas as pd


_ROOT = Path(__file__).resolve().parent
_DEFAULT_INPUT = _ROOT / "zone_to_zone_generalized_cost_all_networks_long.csv"
_DEFAULT_ZONES = (
    _ROOT.parent / "Data" / "2011 RMOW RTM TAZ_zone" / "2011 RMOW RTM TAZ_zone.shp"
)
_DEFAULT_OUTPUT = _ROOT / "analysis" / "mode_cost_difference_maps.gpkg"


def _to_bool(v) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in {"true", "1", "yes", "y"}


def _to_numeric(s: pd.Series) -> pd.Series:
    cleaned = (
        s.astype(str)
        .str.replace(r"[^0-9.\-]+", "", regex=True)
        .replace({"": None, "nan": None, "None": None})
    )
    return pd.to_numeric(cleaned, errors="coerce")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export map layers for mode generalized-cost differences.")
    p.add_argument("--input", type=Path, default=_DEFAULT_INPUT, help="all_networks_long CSV path")
    p.add_argument("--zones", type=Path, default=_DEFAULT_ZONES, help="TAZ polygons path (shp/gpkg)")
    p.add_argument("--zone-column", default="ID_TAZ", help="Zone ID field in polygons (default: ID_TAZ)")
    p.add_argument("--out", type=Path, default=_DEFAULT_OUTPUT, help="Output GeoPackage path")
    return p.parse_args()


def _mode_delta_tables(df: pd.DataFrame) -> pd.DataFrame:
    od_cols = ["origin_zone_id", "dest_zone_id"]
    wide = (
        df.pivot_table(
            index=od_cols,
            columns="network",
            values="generalized_cost_num",
            aggfunc="min",
        )
        .reset_index()
    )
    for c in ("road", "bus", "ion"):
        if c not in wide.columns:
            wide[c] = pd.NA

    wide["delta_bus_minus_road"] = wide["bus"] - wide["road"]
    wide["delta_ion_minus_road"] = wide["ion"] - wide["road"]
    wide["delta_bus_minus_ion"] = wide["bus"] - wide["ion"]
    return wide


def _zone_summary(wide: pd.DataFrame, delta_col: str) -> pd.DataFrame:
    g = wide.dropna(subset=[delta_col]).groupby("origin_zone_id", as_index=False)[delta_col].agg(
        n_od_pairs="count",
        mean_delta="mean",
        median_delta="median",
        p90_delta=lambda s: float(s.quantile(0.9)),
    )
    g["share_mode1_better"] = (
        wide.dropna(subset=[delta_col])
        .groupby("origin_zone_id")[delta_col]
        .apply(lambda s: float((s < 0).mean()))
        .reindex(g["origin_zone_id"])
        .values
    )
    return g


def main() -> None:
    args = _parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(f"Missing input CSV: {args.input}")
    if not args.zones.is_file():
        raise FileNotFoundError(f"Missing zones file: {args.zones}")

    df = pd.read_csv(args.input)
    req = {"network", "origin_zone_id", "dest_zone_id", "generalized_cost", "found"}
    missing = req.difference(df.columns)
    if missing:
        raise KeyError(f"Input missing columns: {sorted(missing)}")

    df = df.copy()
    df["found"] = df["found"].map(_to_bool)
    df["generalized_cost_num"] = _to_numeric(df["generalized_cost"])
    df = df[(df["found"]) & (df["generalized_cost_num"].notna())].copy()
    if df.empty:
        raise ValueError("No successful rows with numeric generalized_cost.")

    wide = _mode_delta_tables(df)

    zones = gpd.read_file(args.zones)
    if args.zone_column not in zones.columns:
        raise KeyError(f"Zone column {args.zone_column!r} not found in zones file.")
    zones[args.zone_column] = pd.to_numeric(zones[args.zone_column], errors="coerce")
    zones = zones.dropna(subset=[args.zone_column]).copy()
    zones[args.zone_column] = zones[args.zone_column].astype(int)
    zones_wgs = zones.to_crs("EPSG:4326")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        args.out.unlink()

    delta_specs = [
        ("delta_bus_minus_road", "zone_delta_bus_minus_road"),
        ("delta_ion_minus_road", "zone_delta_ion_minus_road"),
        ("delta_bus_minus_ion", "zone_delta_bus_minus_ion"),
    ]

    for delta_col, zone_layer in delta_specs:
        zone_stats = _zone_summary(wide, delta_col)
        z = zones_wgs.merge(zone_stats, left_on=args.zone_column, right_on="origin_zone_id", how="left")
        z = z[z["origin_zone_id"].notna()].copy()
        z.to_file(args.out, layer=zone_layer, driver="GPKG")

    print("Mode-difference map layers exported:")
    print(f"  {args.out.resolve()}")
    print("Layers:")
    for _, zone_layer in delta_specs:
        print(f"  - {zone_layer}")


if __name__ == "__main__":
    main()
