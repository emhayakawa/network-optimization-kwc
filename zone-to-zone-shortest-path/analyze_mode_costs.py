"""
Post-process zone-to-zone generalized costs across road, bus, and ION.

Usage:
    python zone-to-zone-shortest-path/analyze_mode_costs.py

Default input:
    zone-to-zone-shortest-path/zone_to_zone_generalized_cost_all_networks_long.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


_ROOT = Path(__file__).resolve().parent
_DEFAULT_INPUT = _ROOT / "zone_to_zone_generalized_cost_all_networks_long.csv"
_DEFAULT_OUTDIR = _ROOT / "analysis"


def _to_bool(v) -> bool:
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    return s in {"true", "1", "yes", "y"}


def _to_numeric_series(s: pd.Series) -> pd.Series:
    # Handles "$12.34" style values and plain numeric strings.
    cleaned = (
        s.astype(str)
        .str.replace(r"[^0-9.\-]+", "", regex=True)
        .replace({"": None, "nan": None, "None": None})
    )
    return pd.to_numeric(cleaned, errors="coerce")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze zone-to-zone mode generalized costs.")
    p.add_argument("--input", type=Path, default=_DEFAULT_INPUT, help="Input all_networks_long CSV path.")
    p.add_argument("--out-dir", type=Path, default=_DEFAULT_OUTDIR, help="Output directory for analysis files.")
    p.add_argument(
        "--high-transit-quantile",
        type=float,
        default=0.95,
        help="Quantile threshold for high-transit-cost OD pairs (default: 0.95).",
    )
    p.add_argument(
        "--distance-bins",
        type=int,
        default=8,
        help="Number of quantile bins for road distance trend summaries (default: 8).",
    )
    p.add_argument(
        "--max-scatter-points",
        type=int,
        default=30000,
        help="Max points per scatter plot (default: 30000).",
    )
    p.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip creating PNG visualizations.",
    )
    return p.parse_args()


def _plot_mode_distribution(ok: pd.DataFrame, fig_dir: Path) -> None:
    plt.figure(figsize=(8, 5))
    order = sorted(ok["network"].dropna().unique().tolist())
    data = [ok.loc[ok["network"] == n, "generalized_cost_num"].dropna().values for n in order]
    plt.boxplot(data, tick_labels=order, showfliers=False)
    plt.ylabel("Generalized cost")
    plt.title("Generalized Cost Distribution by Network")
    plt.tight_layout()
    plt.savefig(fig_dir / "gc_distribution_by_network.png", dpi=160)
    plt.close()


def _plot_scatter_by_network(
    ok: pd.DataFrame,
    x_col: str,
    y_col: str,
    xlabel: str,
    ylabel: str,
    title: str,
    out_path: Path,
    max_points: int,
) -> None:
    sub = ok[[x_col, y_col, "network"]].dropna()
    if sub.empty:
        return
    if len(sub) > max_points:
        sub = sub.sample(n=max_points, random_state=42)

    colors = {"road": "#1f77b4", "bus": "#ff7f0e", "ion": "#2ca02c"}
    plt.figure(figsize=(8, 5))
    for n in sorted(sub["network"].unique().tolist()):
        d = sub[sub["network"] == n]
        plt.scatter(d[x_col], d[y_col], s=7, alpha=0.35, label=n, c=colors.get(n, None))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def _plot_gc_vs_transfers(ok: pd.DataFrame, fig_dir: Path) -> None:
    sub = ok[ok["network"].isin(["bus", "ion"])].copy()
    if sub.empty:
        return
    sub["num_transfers_num"] = _to_numeric_series(sub.get("num_transfers", pd.Series(index=sub.index)))
    sub = sub.dropna(subset=["num_transfers_num", "generalized_cost_num"])
    if sub.empty:
        return
    sub["num_transfers_num"] = sub["num_transfers_num"].astype(int).clip(lower=0, upper=6)
    grp = (
        sub.groupby(["network", "num_transfers_num"], as_index=False)["generalized_cost_num"]
        .mean()
        .sort_values(["network", "num_transfers_num"])
    )
    plt.figure(figsize=(8, 5))
    for n in sorted(grp["network"].unique().tolist()):
        d = grp[grp["network"] == n]
        plt.plot(d["num_transfers_num"], d["generalized_cost_num"], marker="o", label=n)
    plt.xlabel("Number of transfers (capped at 6)")
    plt.ylabel("Average generalized cost")
    plt.title("Transit Generalized Cost vs Transfers")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(fig_dir / "gc_vs_num_transfers_bus_ion.png", dpi=160)
    plt.close()


def _plot_mode_advantage_by_distance(trend: pd.DataFrame, fig_dir: Path) -> None:
    cols = {"road", "bus", "ion", "road_length_km"}
    if not cols.issubset(trend.columns):
        return
    d = trend.dropna(subset=["road", "road_length_km"]).copy()
    if d.empty:
        return
    bins = min(10, max(3, int(d["road_length_km"].nunique())))
    d["dist_bin"] = pd.qcut(d["road_length_km"], q=bins, duplicates="drop")
    g = d.groupby("dist_bin", as_index=False).agg(
        avg_road_km=("road_length_km", "mean"),
        avg_road_gc=("road", "mean"),
        avg_bus_gc=("bus", "mean"),
        avg_ion_gc=("ion", "mean"),
    )
    plt.figure(figsize=(8, 5))
    plt.plot(g["avg_road_km"], g["avg_road_gc"], marker="o", label="road")
    plt.plot(g["avg_road_km"], g["avg_bus_gc"], marker="o", label="bus")
    plt.plot(g["avg_road_km"], g["avg_ion_gc"], marker="o", label="ion")
    plt.xlabel("Average road distance in bin (km)")
    plt.ylabel("Average generalized cost")
    plt.title("Generalized Cost vs Trip Distance")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(fig_dir / "gc_by_distance_bin_mode_comparison.png", dpi=160)
    plt.close()


def main() -> None:
    args = _parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)
    required = {"network", "origin_zone_id", "dest_zone_id", "found", "generalized_cost"}
    missing = required.difference(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    df = df.copy()
    df["found"] = df["found"].map(_to_bool)
    df["generalized_cost_num"] = _to_numeric_series(df["generalized_cost"])
    if "total_length_m" in df.columns:
        df["total_length_m_num"] = _to_numeric_series(df["total_length_m"])
    else:
        df["total_length_m_num"] = pd.NA
    if "total_time_min" in df.columns:
        df["total_time_min_num"] = _to_numeric_series(df["total_time_min"])
    else:
        df["total_time_min_num"] = pd.NA
    if "num_transfers" in df.columns:
        df["num_transfers_num"] = _to_numeric_series(df["num_transfers"])
    else:
        df["num_transfers_num"] = pd.NA

    # Keep successful paths with valid generalized cost.
    ok = df[(df["found"]) & (df["generalized_cost_num"].notna())].copy()
    if ok.empty:
        raise ValueError("No successful rows with numeric generalized_cost found.")

    # 1) Average generalized cost by mode/network.
    avg = (
        ok.groupby("network", as_index=False)
        .agg(
            od_pairs=("generalized_cost_num", "count"),
            avg_generalized_cost=("generalized_cost_num", "mean"),
            median_generalized_cost=("generalized_cost_num", "median"),
            p90_generalized_cost=("generalized_cost_num", lambda s: float(s.quantile(0.9))),
        )
        .sort_values("network")
    )
    avg.to_csv(args.out_dir / "average_generalized_cost_by_network.csv", index=False)

    # Pivot to OD-level mode comparison tables.
    od_cols = ["origin_zone_id", "dest_zone_id"]
    gc_wide = (
        ok.pivot_table(
            index=od_cols,
            columns="network",
            values="generalized_cost_num",
            aggfunc="min",
        )
        .reset_index()
    )
    len_wide = (
        ok.pivot_table(
            index=od_cols,
            columns="network",
            values="total_length_m_num",
            aggfunc="min",
        )
        .reset_index()
    )
    # 2) Comparison summaries.
    def _pair_summary(a: str, b: str) -> dict:
        if a not in gc_wide.columns or b not in gc_wide.columns:
            return {"pair": f"{a}_vs_{b}", "available": False}
        sub = gc_wide[[a, b]].dropna()
        if sub.empty:
            return {"pair": f"{a}_vs_{b}", "available": False}
        diff = sub[a] - sub[b]
        return {
            "pair": f"{a}_vs_{b}",
            "available": True,
            "n_overlap_od": int(len(sub)),
            "share_a_better": float((diff < 0).mean()),
            "share_b_better": float((diff > 0).mean()),
            "share_tie": float((diff == 0).mean()),
            "avg_delta_a_minus_b": float(diff.mean()),
            "median_delta_a_minus_b": float(diff.median()),
        }

    pair_summaries = [
        _pair_summary("road", "bus"),
        _pair_summary("road", "ion"),
        _pair_summary("bus", "ion"),
    ]
    pd.DataFrame(pair_summaries).to_csv(args.out_dir / "pairwise_mode_comparison_summary.csv", index=False)

    # 3) Distance-binned trends (road distance as x-axis).
    if "road" in len_wide.columns and "road" in gc_wide.columns:
        road_len = len_wide[od_cols + ["road"]].rename(columns={"road": "road_length_m"})
        trend = gc_wide.merge(road_len, on=od_cols, how="left")
        trend = trend[trend["road_length_m"].notna()].copy()
        if not trend.empty:
            trend["road_length_km"] = trend["road_length_m"] / 1000.0
            q = max(2, int(args.distance_bins))
            trend["distance_bin"] = pd.qcut(
                trend["road_length_km"],
                q=min(q, trend["road_length_km"].nunique()),
                duplicates="drop",
            )

            out = (
                trend.groupby("distance_bin", as_index=False)
                .agg(
                    od_pairs=("road_length_km", "count"),
                    avg_road_km=("road_length_km", "mean"),
                    avg_road_gc=("road", "mean"),
                    avg_bus_gc=("bus", "mean"),
                    avg_ion_gc=("ion", "mean"),
                    share_bus_better_than_road=("bus", lambda s: float((s < trend.loc[s.index, "road"]).mean())),
                    share_ion_better_than_road=("ion", lambda s: float((s < trend.loc[s.index, "road"]).mean())),
                )
            )
            out.to_csv(args.out_dir / "road_vs_transit_by_distance_bin.csv", index=False)
    else:
        trend = pd.DataFrame()

    # 4) OD pairs with extremely high transit cost.
    transit_cols = [c for c in ("bus", "ion") if c in gc_wide.columns]
    if transit_cols:
        x = gc_wide[od_cols + transit_cols].copy()
        x["transit_min_gc"] = x[transit_cols].min(axis=1, skipna=True)
        x = x[x["transit_min_gc"].notna()].copy()
        if not x.empty:
            qv = float(x["transit_min_gc"].quantile(args.high_transit_quantile))
            high = x[x["transit_min_gc"] >= qv].sort_values("transit_min_gc", ascending=False)
            high.to_csv(args.out_dir / "extreme_high_transit_cost_od_pairs.csv", index=False)

            meta = {
                "input_file": str(args.input.resolve()),
                "high_transit_quantile": float(args.high_transit_quantile),
                "high_transit_threshold": qv,
                "n_high_transit_od_pairs": int(len(high)),
                "outputs": [
                    "average_generalized_cost_by_network.csv",
                    "pairwise_mode_comparison_summary.csv",
                    "road_vs_transit_by_distance_bin.csv",
                    "extreme_high_transit_cost_od_pairs.csv",
                ],
            }
            (args.out_dir / "analysis_summary.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    if not args.no_plots:
        fig_dir = args.out_dir / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)
        _plot_mode_distribution(ok, fig_dir)
        _plot_scatter_by_network(
            ok,
            x_col="total_length_m_num",
            y_col="generalized_cost_num",
            xlabel="Trip length (m)",
            ylabel="Generalized cost",
            title="Generalized Cost vs Distance by Network",
            out_path=fig_dir / "gc_vs_distance_by_network.png",
            max_points=args.max_scatter_points,
        )
        _plot_scatter_by_network(
            ok,
            x_col="total_time_min_num",
            y_col="generalized_cost_num",
            xlabel="In-vehicle time (min)",
            ylabel="Generalized cost",
            title="Generalized Cost vs Travel Time by Network",
            out_path=fig_dir / "gc_vs_time_by_network.png",
            max_points=args.max_scatter_points,
        )
        _plot_gc_vs_transfers(ok, fig_dir)
        if "trend" in locals() and not trend.empty:
            _plot_mode_advantage_by_distance(trend, fig_dir)

    print("Analysis complete.")
    print(f"Input:  {args.input.resolve()}")
    print(f"Output: {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
