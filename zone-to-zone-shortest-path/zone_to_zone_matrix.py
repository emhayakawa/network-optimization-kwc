"""
Build full zone-to-zone generalized-cost matrices for road, bus, and ION.

For each network and each (origin_zone, destination_zone) pair, this script uses:

- Transit (bus, ion): ``shortest_path_transit_zone_to_zone(..., cost="generalized")``
- Road: ``build_road_zone_to_zone_matrix`` — one multi-source Dijkstra per origin zone, then
  scans all destination zones (see ``zone_to_zone_routing.py``).

Default usage (from repo root):

    python zone-to-zone-shortest-path/zone_to_zone_matrix.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Iterable, List, Optional, Tuple

import pandas as pd

_ROOT = Path(__file__).resolve().parent
_REPO_ROOT = _ROOT.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from transit.shortest_path import COST_GENERALIZED, build_transit_runtime_cache
from road_network.shortest_path import build_networkx_graph
from zone_to_zone_routing import (
    build_road_zone_to_zone_matrix,
    shortest_path_transit_zone_to_zone,
)


_DEFAULT_ZONES_FILE = _REPO_ROOT / "Data" / "ion_buffer_zones.xlsx"
_DEFAULT_ROAD_NODES = _REPO_ROOT / "road_network" / "data" / "gmns" / "node.csv"
_DEFAULT_ROAD_LINKS = _REPO_ROOT / "road_network" / "data" / "gmns" / "link.csv"
_DEFAULT_BUS_NODES = _REPO_ROOT / "bus_network" / "data" / "node.csv"
_DEFAULT_BUS_LINKS = _REPO_ROOT / "bus_network" / "data" / "link.csv"
_DEFAULT_ION_NODES = _REPO_ROOT / "ion_network" / "data" / "node.csv"
_DEFAULT_ION_LINKS = _REPO_ROOT / "ion_network" / "data" / "link.csv"
_DEFAULT_OUT_DIR = _ROOT
_DEFAULT_OUT_PREFIX = "zone_to_zone_generalized_cost"


def _choose_zone_column(
    df: pd.DataFrame,
    user_col: Optional[str],
    zone_col_index: Optional[int],
) -> str:
    if user_col:
        if user_col not in df.columns:
            raise KeyError(
                f"Requested --zone-column {user_col!r} not found. Available columns: {list(df.columns)}"
            )
        return user_col

    if zone_col_index is not None:
        if zone_col_index < 0 or zone_col_index >= len(df.columns):
            raise IndexError(
                f"--zone-col-index={zone_col_index} out of range for columns {list(df.columns)}"
            )
        return str(df.columns[zone_col_index])

    preferred = ("zone_id", "ZONE_ID", "Zone_ID", "zone", "ZONE", "Zone")
    for col in preferred:
        if col in df.columns:
            return col

    # Fallback: first column that can be parsed to at least one numeric value.
    for col in df.columns:
        vals = pd.to_numeric(df[col], errors="coerce")
        if vals.notna().any():
            return str(col)

    raise ValueError(
        "Could not infer a zone column in zones file. "
        "Please pass --zone-column explicitly."
    )


def _read_zones_table(path: Path, sheet_name: Optional[str]) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Zones file not found: {path}")

    suffix = path.suffix.lower()
    if suffix in (".xlsx", ".xls"):
        try:
            return pd.read_excel(path, sheet_name=sheet_name or 0)
        except ImportError as exc:
            raise ImportError(
                "Reading .xlsx requires openpyxl. Install with: pip install openpyxl"
            ) from exc
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(
        f"Unsupported zones file type {suffix!r}. Use .xlsx/.xls or .csv."
    )


def _load_zone_ids(
    path: Path,
    zone_column: Optional[str],
    zone_col_index: Optional[int],
    sheet_name: Optional[str],
) -> List[int]:
    df = _read_zones_table(path, sheet_name)
    zcol = _choose_zone_column(df, zone_column, zone_col_index)
    z = pd.to_numeric(df[zcol], errors="coerce").dropna().astype(int)
    zones = sorted(set(int(v) for v in z.tolist()))
    if not zones:
        raise ValueError(f"No valid numeric zones found in column {zcol!r} from {path}")
    print(f"Loaded {len(zones)} unique zones from {path} (column={zcol!r})")
    return zones


def _ensure_numeric_zone_id(nodes_df: pd.DataFrame) -> pd.DataFrame:
    if "zone_id" not in nodes_df.columns:
        raise KeyError("nodes_df has no 'zone_id' column. Rebuild network with zone assignment enabled.")
    out = nodes_df.copy()
    out["zone_id"] = pd.to_numeric(out["zone_id"], errors="coerce")
    return out


def _zone_overlap_summary(zones: List[int], nodes_df: pd.DataFrame) -> Tuple[int, int]:
    node_zones = set(nodes_df["zone_id"].dropna().astype(int).tolist())
    overlap = sorted(set(zones) & node_zones)
    return len(overlap), len(node_zones)


def _zones_with_nodes(zones: List[int], nodes_df: pd.DataFrame) -> List[int]:
    node_zones = set(nodes_df["zone_id"].dropna().astype(int).tolist())
    return sorted([z for z in zones if z in node_zones])


def _iter_pairs(zones: Iterable[int]):
    zlist = list(zones)
    for o in zlist:
        for d in zlist:
            yield int(o), int(d)


def _reduce_bus_nodes_one_per_route_per_zone(bus_nodes_df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep one representative bus node per (zone_id, route_id).

    Selection rule:
    1) choose the node closest to the zone centroid (centroid from mean x/y of bus nodes in zone),
    2) break ties by smallest node_id.
    """
    required = {"node_id", "zone_id", "route_id"}
    missing = required.difference(bus_nodes_df.columns)
    if missing:
        raise KeyError(
            f"Bus node reduction requires columns {sorted(required)}; missing: {sorted(missing)}"
        )

    nodes = bus_nodes_df.copy()
    nodes["zone_id"] = pd.to_numeric(nodes["zone_id"], errors="coerce")
    nodes["node_id"] = pd.to_numeric(nodes["node_id"], errors="coerce")
    nodes["route_id"] = pd.to_numeric(nodes["route_id"], errors="coerce")
    nodes = nodes.dropna(subset=["zone_id", "node_id"])
    nodes["zone_id"] = nodes["zone_id"].astype(int)
    nodes["node_id"] = nodes["node_id"].astype(int)
    # Keep missing route_id as a dedicated group so those nodes are still eligible.
    nodes["route_group"] = nodes["route_id"].fillna(-1).astype(int)

    has_xy = "x_coord" in nodes.columns and "y_coord" in nodes.columns
    if has_xy:
        nodes["x_coord"] = pd.to_numeric(nodes["x_coord"], errors="coerce")
        nodes["y_coord"] = pd.to_numeric(nodes["y_coord"], errors="coerce")

    keep_idx = []
    for _, zone_group in nodes.groupby("zone_id", sort=False):
        centroid_x = centroid_y = None
        if has_xy:
            xy = zone_group.dropna(subset=["x_coord", "y_coord"])
            if not xy.empty:
                centroid_x = float(xy["x_coord"].mean())
                centroid_y = float(xy["y_coord"].mean())

        for _, route_group in zone_group.groupby("route_group", sort=False):
            cand = route_group.copy()
            if centroid_x is not None and centroid_y is not None:
                dx = cand["x_coord"] - centroid_x
                dy = cand["y_coord"] - centroid_y
                cand["dist2"] = (dx * dx) + (dy * dy)
                cand["dist2"] = cand["dist2"].fillna(float("inf"))
                cand = cand.sort_values(["dist2", "node_id"])
            else:
                cand = cand.sort_values(["node_id"])
            keep_idx.append(int(cand.index[0]))

    reduced = nodes.loc[sorted(set(keep_idx))].drop(columns=["route_group"])
    reduced = reduced.sort_values("node_id").reset_index(drop=True)
    return reduced


def _reduce_ion_nodes_one_per_route_per_zone_keep_ion301(ion_nodes_df: pd.DataFrame) -> pd.DataFrame:
    """
    ION node reduction with exception:
    - default: one representative node per (zone_id, route_id), nearest to zone centroid
      with node_id tie-break.
    - exception: if a zone has any ``route_id == 'ION_301'`` node, keep all nodes in that zone.
    """
    required = {"node_id", "zone_id", "route_id"}
    missing = required.difference(ion_nodes_df.columns)
    if missing:
        raise KeyError(
            f"ION node reduction requires columns {sorted(required)}; missing: {sorted(missing)}"
        )

    nodes = ion_nodes_df.copy()
    nodes["zone_id"] = pd.to_numeric(nodes["zone_id"], errors="coerce")
    nodes["node_id"] = pd.to_numeric(nodes["node_id"], errors="coerce")
    nodes = nodes.dropna(subset=["zone_id", "node_id"])
    nodes["zone_id"] = nodes["zone_id"].astype(int)
    nodes["node_id"] = nodes["node_id"].astype(int)
    nodes["route_id_str"] = nodes["route_id"].astype(str)
    nodes["route_group"] = nodes["route_id_str"].fillna("-1")

    has_xy = "x_coord" in nodes.columns and "y_coord" in nodes.columns
    if has_xy:
        nodes["x_coord"] = pd.to_numeric(nodes["x_coord"], errors="coerce")
        nodes["y_coord"] = pd.to_numeric(nodes["y_coord"], errors="coerce")

    keep_idx = []
    for _, zone_group in nodes.groupby("zone_id", sort=False):
        # Keep all nodes for any zone touched by ION_301.
        if (zone_group["route_id_str"] == "ION_301").any():
            keep_idx.extend(zone_group.index.tolist())
            continue

        centroid_x = centroid_y = None
        if has_xy:
            xy = zone_group.dropna(subset=["x_coord", "y_coord"])
            if not xy.empty:
                centroid_x = float(xy["x_coord"].mean())
                centroid_y = float(xy["y_coord"].mean())

        for _, route_group in zone_group.groupby("route_group", sort=False):
            cand = route_group.copy()
            if centroid_x is not None and centroid_y is not None:
                dx = cand["x_coord"] - centroid_x
                dy = cand["y_coord"] - centroid_y
                cand["dist2"] = (dx * dx) + (dy * dy)
                cand["dist2"] = cand["dist2"].fillna(float("inf"))
                cand = cand.sort_values(["dist2", "node_id"])
            else:
                cand = cand.sort_values(["node_id"])
            keep_idx.append(int(cand.index[0]))

    reduced = nodes.loc[sorted(set(keep_idx))].drop(columns=["route_group", "route_id_str"])
    reduced = reduced.sort_values("node_id").reset_index(drop=True)
    return reduced


def build_zone_matrix_for_transit(
    network_name: str,
    zones: List[int],
    nodes_df: pd.DataFrame,
    links_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    transit_cache = build_transit_runtime_cache(
        nodes_df,
        links_df,
        cost=COST_GENERALIZED,
    )
    matrix = pd.DataFrame(index=zones, columns=zones, dtype=float)
    rows = []
    total = len(zones) * len(zones)
    i = 0
    for o, d in _iter_pairs(zones):
        i += 1
        if i % max(1, total // 20) == 0 or i == total:
            print(f"[{network_name}] Progress: {i}/{total} OD zone pairs")

        res = shortest_path_transit_zone_to_zone(
            nodes_df,
            links_df,
            o,
            d,
            cost=COST_GENERALIZED,
            transit_cache=transit_cache,
            verbose=False,
            list_all_candidates=False,
        )

        gc = float(res["generalized_cost"]) if res.get("found") and res.get("generalized_cost") is not None else float("nan")
        matrix.loc[o, d] = gc

        rows.append(
            {
                "origin_zone_id": o,
                "dest_zone_id": d,
                "found": bool(res.get("found")),
                "generalized_cost": gc,
                "best_origin_node_id": res.get("orig_node_id"),
                "best_dest_node_id": res.get("dest_node_id"),
                "total_time_min": res.get("total_time_min"),
                "total_length_m": res.get("total_length_m"),
                "num_transfers": res.get("num_transfers"),
                "num_multimodal_transfers": res.get("num_multimodal_transfers"),
            }
        )

    long_df = pd.DataFrame(rows)
    long_df.insert(0, "network", network_name)
    return matrix, long_df


def _load_network_csv(nodes_csv: Path, links_csv: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not nodes_csv.is_file() or not links_csv.is_file():
        raise FileNotFoundError(
            f"Missing nodes/links CSV. Expected {nodes_csv} and {links_csv}."
        )
    nodes_df = pd.read_csv(nodes_csv)
    links_df = pd.read_csv(links_csv)
    nodes_df = _ensure_numeric_zone_id(nodes_df)
    return nodes_df, links_df


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute full road/bus/ion zone-to-zone generalized-cost matrices."
    )
    p.add_argument("--zones-file", type=Path, default=_DEFAULT_ZONES_FILE)
    p.add_argument(
        "--zone-column",
        default="ID_TAZ",
        help="Column in zones file containing zone IDs (default: ID_TAZ).",
    )
    p.add_argument(
        "--zone-col-index",
        type=int,
        default=1,
        help="Fallback 0-based zone column index (ignored when --zone-column is set).",
    )
    p.add_argument(
        "--sheet-name",
        default=None,
        help="Excel sheet name (or index string). Default: first sheet.",
    )
    p.add_argument("--road-nodes-csv", type=Path, default=_DEFAULT_ROAD_NODES)
    p.add_argument("--road-links-csv", type=Path, default=_DEFAULT_ROAD_LINKS)
    p.add_argument("--bus-nodes-csv", type=Path, default=_DEFAULT_BUS_NODES)
    p.add_argument("--bus-links-csv", type=Path, default=_DEFAULT_BUS_LINKS)
    p.add_argument("--ion-nodes-csv", type=Path, default=_DEFAULT_ION_NODES)
    p.add_argument("--ion-links-csv", type=Path, default=_DEFAULT_ION_LINKS)
    p.add_argument("--out-dir", type=Path, default=_DEFAULT_OUT_DIR)
    p.add_argument(
        "--out-prefix",
        default=_DEFAULT_OUT_PREFIX,
        help="Output filename prefix (default: zone_to_zone_generalized_cost).",
    )
    p.add_argument(
        "--network",
        choices=("all", "road", "bus", "ion"),
        default="all",
        help="Which network(s) to run: all|road|bus|ion (default: all).",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    zones = _load_zone_ids(
        args.zones_file,
        args.zone_column,
        args.zone_col_index,
        args.sheet_name,
    )

    selected = ("road", "bus", "ion") if args.network == "all" else (args.network,)
    network_data: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    if "road" in selected:
        network_data["road"] = _load_network_csv(args.road_nodes_csv, args.road_links_csv)
    if "bus" in selected:
        network_data["bus"] = _load_network_csv(args.bus_nodes_csv, args.bus_links_csv)
    if "ion" in selected:
        network_data["ion"] = _load_network_csv(args.ion_nodes_csv, args.ion_links_csv)

    overlaps = {}
    for name in selected:
        ndf = network_data[name][0]
        ov, nz = _zone_overlap_summary(zones, ndf)
        overlaps[name] = ov
        print(f"Zone overlap [{name}]: {ov} overlapping zones (node zone count={nz})")

    if all(overlaps[name] == 0 for name in selected):
        raise ValueError(
            "No overlap between zones file and any network zone_id values. "
            "Check --zone-column/--zone-col-index and zone coding."
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    zones_by_network: dict[str, list[int]] = {}
    for name in selected:
        zones_by_network[name] = _zones_with_nodes(zones, network_data[name][0])

    print("\nRunning OD only on zones with nodes:")
    for name in selected:
        print(f"  {name:<4}: {len(zones_by_network[name])} zones")

    matrices: dict[str, pd.DataFrame] = {}
    longs: dict[str, pd.DataFrame] = {}
    if "road" in selected:
        road_nodes, road_links = network_data["road"]
        road_graph = build_networkx_graph(road_links, "generalized_cost")
        matrices["road"], longs["road"] = build_road_zone_to_zone_matrix(
            road_nodes,
            road_links,
            zones_by_network["road"],
            weight_field="generalized_cost",
            graph=road_graph,
            network_name="road",
        )
    if "bus" in selected:
        bus_nodes, bus_links = network_data["bus"]
        bus_nodes_reduced = _reduce_bus_nodes_one_per_route_per_zone(bus_nodes)
        print(
            f"\n[bus] Representative-node rule active: "
            f"{len(bus_nodes)} -> {len(bus_nodes_reduced)} candidate OD nodes "
            "(one per zone+route, nearest to zone centroid)"
        )
        matrices["bus"], longs["bus"] = build_zone_matrix_for_transit(
            "bus", zones_by_network["bus"], bus_nodes_reduced, bus_links
        )
    if "ion" in selected:
        ion_nodes, ion_links = network_data["ion"]
        ion_nodes_reduced = _reduce_ion_nodes_one_per_route_per_zone_keep_ion301(ion_nodes)
        print(
            f"\n[ion] Representative-node rule active with ION_301 exception: "
            f"{len(ion_nodes)} -> {len(ion_nodes_reduced)} candidate OD nodes"
        )
        matrices["ion"], longs["ion"] = build_zone_matrix_for_transit(
            "ion", zones_by_network["ion"], ion_nodes_reduced, ion_links
        )

    path_long_all = args.out_dir / f"{args.out_prefix}_all_networks_long.csv"
    output_paths = []
    for name in selected:
        path_matrix = args.out_dir / f"{args.out_prefix}_{name}_matrix.csv"
        matrices[name].to_csv(path_matrix, index=True)
        output_paths.append(path_matrix)
    long_all = pd.concat([longs[name] for name in selected], ignore_index=True)
    long_all.to_csv(path_long_all, index=False)
    output_paths.append(path_long_all)

    for name in selected:
        sub = long_all[long_all["network"] == name]
        found_n = int(sub["found"].sum())
        print(f"\n{name}: Found paths for {found_n}/{len(sub)} OD zone pairs.")
    print("\nOutput files:")
    for p in output_paths:
        print(f"  {p.resolve()}")


if __name__ == "__main__":
    main()
