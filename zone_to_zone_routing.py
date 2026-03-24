"""
Zone-to-zone shortest path on top of existing node-to-node engines.

Node-to-node routing is unchanged in ``transit.shortest_path`` and ``road_network.shortest_path``.
This module only enumerates OD pairs (all nodes in origin TAZ × all in destination TAZ) and keeps
the best path by cost.

- **Transit (bus-only or ION multimodal):** same inputs as ``shortest_path_transit``.
- **Road:** NetworkX Dijkstra on ``edges_gdf`` (same as ``compute_shortest_path_networkx``).

Run from the repo root: ``python zone_to_zone_routing.py`` (defaults from ``zone_to_zone_config.py``),
or ``python zone_to_zone_routing.py --help`` for CLI overrides.
"""
from __future__ import annotations

import os
import sys

import pandas as pd

_URA_ROOT = os.path.dirname(os.path.abspath(__file__))
if _URA_ROOT not in sys.path:
    sys.path.insert(0, _URA_ROOT)

from transit.shortest_path import (
    COST_DISTANCE,
    COST_GENERALIZED,
    COST_TIME,
    shortest_path_transit,
)


def _print_transit_candidate_paths(
    candidates,
    cost,
    orig_zone_id,
    dest_zone_id,
    best_pair,
):
    """``candidates``: list of (score, orig_node, dest_node, res_dict)."""
    if not candidates:
        return
    ranked = sorted(candidates, key=lambda x: x[0])
    if cost == COST_GENERALIZED:
        obj_label = "gen_cost_$"
    elif cost == COST_TIME:
        obj_label = "ivt_min (obj)"
    else:
        obj_label = "length_m (obj)"

    print(
        f"\n--- All successful OD pairs (sorted by {cost}; * = chosen shortest) ---"
    )
    show_gc = cost != COST_GENERALIZED
    hdr = (
        f"  {'#':>4}  {'orig_n':>8}  {'dest_n':>8}  {obj_label:>14}  "
        f"{'ivt_min':>10}  {'len_m':>10}  "
    )
    hdr += f"{'gen_$':>10}  " if show_gc else ""
    hdr += f"{'xfer':>5}"
    print(hdr)
    for i, (s, o, d, res) in enumerate(ranked, start=1):
        mark = " *" if (o, d) == best_pair else "  "
        ivt = res.get("total_time_min")
        ln = res.get("total_length_m")
        gc = res.get("generalized_cost")
        xf = res.get("num_transfers")
        ivt_s = f"{ivt:.2f}" if ivt is not None and not pd.isna(ivt) else "-"
        ln_s = f"{ln:.0f}" if ln is not None and not pd.isna(ln) else "-"
        gc_s = f"{gc:.2f}" if gc is not None and not pd.isna(gc) else "-"
        try:
            xf_s = str(int(float(xf))) if xf is not None and not pd.isna(xf) else "-"
        except (TypeError, ValueError):
            xf_s = "-"
        row = (
            f"  {i:4d}{mark}  {o:8d}  {d:8d}  {s:14.4f}  "
            f"{ivt_s:>10}  {ln_s:>10}  "
        )
        row += f"{gc_s:>10}  " if show_gc else ""
        row += f"{xf_s:>5}"
        print(row)
    print(
        f"\n  >>> Shortest by {cost}: nodes {best_pair[0]} -> {best_pair[1]} "
        f"(zones {orig_zone_id} -> {dest_zone_id})"
    )


def _print_road_candidate_paths(candidates, weight_field, orig_zone_id, dest_zone_id, best_pair):
    if not candidates:
        return
    ranked = sorted(candidates, key=lambda x: x[0])
    print(
        f"\n--- All successful OD pairs (sorted by {weight_field}; * = chosen shortest) ---"
    )
    print(f"  {'#':>4}  {'orig_n':>8}  {'dest_n':>8}  {weight_field:>16}  {'n_nodes':>8}")
    for i, (cst, o, d, res) in enumerate(ranked, start=1):
        mark = " *" if (o, d) == best_pair else "  "
        pn = res.get("path_nodes") or []
        nn = len(pn) if pn else 0
        print(f"  {i:4d}{mark}  {o:8d}  {d:8d}  {cst:16.6f}  {nn:8d}")
    print(
        f"\n  >>> Shortest by {weight_field}: nodes {best_pair[0]} -> {best_pair[1]} "
        f"(zones {orig_zone_id} -> {dest_zone_id})"
    )


def node_ids_in_zone(nodes, zone_id):
    """
    Return ``node_id`` values whose ``zone_id`` matches (requires ``zone_id`` column).

    ``nodes`` may be a pandas DataFrame or a GeoDataFrame with ``node_id`` and ``zone_id``.
    """
    if "zone_id" not in nodes.columns or "node_id" not in nodes.columns:
        return []
    z = int(zone_id)
    return nodes[nodes["zone_id"].astype(int) == z]["node_id"].astype(int).tolist()


def shortest_path_transit_zone_to_zone(
    nodes_df,
    links_df,
    orig_zone_id,
    dest_zone_id,
    cost="generalized",
    fare=None,
    waiting_time_min=None,
    value_of_time=None,
    verbose=True,
    list_all_candidates=False,
):
    """
    Best path between two TAZs on a bus or ION (multimodal) link table.

    Minimizes the same objective as ``shortest_path_transit`` over every node pair
    (origin zone × destination zone). O(|O|·|D|) Dijkstra runs.

    If ``list_all_candidates`` and ``verbose``, prints every successful OD pair sorted
    by the objective, with ``*`` on the chosen shortest.
    """
    orig_nodes = node_ids_in_zone(nodes_df, orig_zone_id)
    dest_nodes = node_ids_in_zone(nodes_df, dest_zone_id)

    empty = {
        "path_nodes": None,
        "path_links": None,
        "total_time_min": None,
        "total_length_m": None,
        "generalized_cost": None,
        "num_transfers": None,
        "num_multimodal_transfers": None,
        "modes_used": None,
        "routes_used": None,
        "found": False,
        "orig_node_id": None,
        "dest_node_id": None,
    }

    if not orig_nodes or not dest_nodes:
        if verbose:
            print(
                f"No nodes in zone(s): orig_zone={orig_zone_id} ({len(orig_nodes)} nodes), "
                f"dest_zone={dest_zone_id} ({len(dest_nodes)} nodes). "
                "Build the network with assign_zone_id=True and a TAZ shapefile."
            )
        return empty

    if cost == COST_GENERALIZED:

        def score(res):
            g = res.get("generalized_cost")
            return float("inf") if g is None or pd.isna(g) else float(g)

    elif cost == COST_TIME:

        def score(res):
            t = res.get("total_time_min")
            return float("inf") if t is None or pd.isna(t) else float(t)

    else:

        def score(res):
            m = res.get("total_length_m")
            return float("inf") if m is None or pd.isna(m) else float(m)

    best = None
    best_score = float("inf")
    best_pair = (None, None)
    candidates = []
    n_grid = len(orig_nodes) * len(dest_nodes)

    for o in orig_nodes:
        for d in dest_nodes:
            res = shortest_path_transit(
                nodes_df,
                links_df,
                o,
                d,
                cost=cost,
                fare=fare,
                waiting_time_min=waiting_time_min,
                value_of_time=value_of_time,
                verbose=False,
            )
            if not res.get("found"):
                continue
            s = score(res)
            if list_all_candidates:
                candidates.append((s, o, d, dict(res)))
            if s < best_score:
                best_score = s
                best = res
                best_pair = (o, d)

    if best is None:
        if verbose:
            print(
                f"No path found between zone {orig_zone_id} and zone {dest_zone_id} "
                f"({len(orig_nodes)}×{len(dest_nodes)} = {n_grid} OD pairs checked, 0 with path)"
            )
        return empty

    best = dict(best)
    best["orig_node_id"] = best_pair[0]
    best["dest_node_id"] = best_pair[1]

    if verbose:
        n_ok = len(candidates) if list_all_candidates else None
        if list_all_candidates and n_ok is not None:
            print(
                f"Zone-to-zone transit sweep: zones {orig_zone_id} -> {dest_zone_id} | "
                f"|O|={len(orig_nodes)} |D|={len(dest_nodes)} | "
                f"grid={n_grid} | paths found={n_ok} | no path={n_grid - n_ok}"
            )
            _print_transit_candidate_paths(
                candidates, cost, orig_zone_id, dest_zone_id, best_pair
            )
        print(
            f"Zone-to-zone transit ({cost}): zone {orig_zone_id} -> {dest_zone_id} "
            f"via nodes {best_pair[0]} -> {best_pair[1]}"
        )
        if cost == COST_GENERALIZED and best.get("generalized_cost") is not None:
            print(f"  Best generalized cost: ${best['generalized_cost']:.2f}")
        elif cost == COST_TIME and best.get("total_time_min") is not None:
            print(f"  Best in-vehicle time: {best['total_time_min']:.2f} min")
        elif best.get("total_length_m") is not None:
            print(f"  Best distance: {best['total_length_m']:.0f} m")

    return best


# Alias: ION uses the same node/link schema as transit shortest path
shortest_path_ion_zone_to_zone = shortest_path_transit_zone_to_zone

# Backward-compatible name used in earlier refactors
shortest_path_zone_to_zone = shortest_path_transit_zone_to_zone


def shortest_path_road_zone_to_zone(
    nodes_gdf,
    edges_gdf,
    orig_zone_id,
    dest_zone_id,
    weight_field="generalized_cost",
    verbose=True,
    list_all_candidates=False,
):
    """
    Best path between two TAZs on the road graph (NetworkX + ``weight_field``).

    Returns:
        dict with path_nodes, path_links, total_cost, found, orig_node_id, dest_node_id

    If ``list_all_candidates`` and ``verbose``, prints every successful OD pair sorted by
    ``weight_field``, with ``*`` on the chosen shortest.
    """
    from road_network.shortest_path import compute_shortest_path_networkx

    orig_nodes = node_ids_in_zone(nodes_gdf, orig_zone_id)
    dest_nodes = node_ids_in_zone(nodes_gdf, dest_zone_id)

    empty = {
        "path_nodes": None,
        "path_links": None,
        "total_cost": None,
        "found": False,
        "orig_node_id": None,
        "dest_node_id": None,
        "weight_field": weight_field,
    }

    if not orig_nodes or not dest_nodes:
        if verbose:
            print(
                f"No road nodes in zone(s): orig_zone={orig_zone_id} ({len(orig_nodes)}), "
                f"dest_zone={dest_zone_id} ({len(dest_nodes)}). "
                "Build roads with assign_zone_id=True."
            )
        return empty

    best = None
    best_cost = float("inf")
    best_pair = (None, None)
    candidates = []
    n_grid = len(orig_nodes) * len(dest_nodes)

    for o in orig_nodes:
        for d in dest_nodes:
            res = compute_shortest_path_networkx(
                edges_gdf, o, d, weight_field=weight_field, verbose=False
            )
            if not res.get("found"):
                continue
            c = float(res["total_cost"])
            if list_all_candidates:
                candidates.append((c, o, d, dict(res)))
            if c < best_cost:
                best_cost = c
                best = res
                best_pair = (o, d)

    if best is None:
        if verbose:
            print(
                f"No road path between zone {orig_zone_id} and {dest_zone_id} "
                f"({len(orig_nodes)}×{len(dest_nodes)} = {n_grid} pairs checked, 0 with path)"
            )
        return empty

    out = {
        "path_nodes": best["path_nodes"],
        "path_links": best["path_links"],
        "total_cost": best["total_cost"],
        "found": True,
        "orig_node_id": best_pair[0],
        "dest_node_id": best_pair[1],
        "weight_field": weight_field,
    }

    if verbose:
        if list_all_candidates and candidates:
            print(
                f"Zone-to-zone road sweep: zones {orig_zone_id} -> {dest_zone_id} | "
                f"|O|={len(orig_nodes)} |D|={len(dest_nodes)} | "
                f"grid={n_grid} | paths found={len(candidates)} | no path={n_grid - len(candidates)}"
            )
            _print_road_candidate_paths(
                candidates, weight_field, orig_zone_id, dest_zone_id, best_pair
            )
        print(
            f"Zone-to-zone road ({weight_field}): zone {orig_zone_id} -> {dest_zone_id} "
            f"via nodes {best_pair[0]} -> {best_pair[1]}"
        )
        print(f"  Best total_cost: {out['total_cost']:.4f}")

    return out


def _load_zone_to_zone_config():
    """Return ``zone_to_zone_config`` module or None if missing."""
    try:
        import zone_to_zone_config as cfg  # noqa: WPS433 — local user config at repo root

        return cfg
    except ImportError:
        return None


if __name__ == "__main__":
    import argparse

    cfg = _load_zone_to_zone_config()

    def _c(name, fallback):
        return getattr(cfg, name, fallback) if cfg is not None else fallback

    parser = argparse.ArgumentParser(
        description=(
            "Zone-to-zone shortest path. Defaults come from zone_to_zone_config.py "
            "(CLI args override)."
        )
    )
    parser.add_argument(
        "--network",
        choices=("bus", "ion", "road"),
        default=_c("NETWORK", "bus"),
        help="Which built network to load (default: zone_to_zone_config.NETWORK)",
    )
    parser.add_argument(
        "--orig",
        type=int,
        default=_c("ORIG_ZONE_ID", 22001),
        help="Origin zone_id (default: zone_to_zone_config.ORIG_ZONE_ID)",
    )
    parser.add_argument(
        "--dest",
        type=int,
        default=_c("DEST_ZONE_ID", 21903),
        help="Destination zone_id (default: zone_to_zone_config.DEST_ZONE_ID)",
    )
    _tc = _c("TRANSIT_COST", COST_GENERALIZED)
    if _tc not in (COST_GENERALIZED, COST_TIME, COST_DISTANCE):
        raise SystemExit(
            f"zone_to_zone_config.TRANSIT_COST must be one of "
            f"{COST_GENERALIZED!r}, {COST_TIME!r}, {COST_DISTANCE!r}; got {_tc!r}"
        )
    parser.add_argument(
        "--cost",
        default=_tc,
        choices=(COST_GENERALIZED, COST_TIME, COST_DISTANCE),
        help="Transit objective (bus/ion); default: zone_to_zone_config.TRANSIT_COST",
    )
    parser.add_argument(
        "--weight-field",
        default=_c("ROAD_WEIGHT_FIELD", "generalized_cost"),
        help="Road link weight; default: zone_to_zone_config.ROAD_WEIGHT_FIELD",
    )
    parser.add_argument(
        "--list-paths",
        action="store_true",
        help="Print every successful OD pair sorted by cost (* = chosen); overrides config",
    )
    parser.add_argument(
        "--no-list-paths",
        action="store_true",
        help="Skip printing all OD pairs (overrides config)",
    )
    args = parser.parse_args()

    if args.list_paths and args.no_list_paths:
        parser.error("Use at most one of --list-paths / --no-list-paths")
    if args.list_paths:
        list_all = True
    elif args.no_list_paths:
        list_all = False
    else:
        list_all = _c("LIST_ALL_CANDIDATE_PATHS", False)

    if args.network in ("bus", "ion"):
        sub = "bus_network" if args.network == "bus" else "ion_network"
        default_node = os.path.join(_URA_ROOT, sub, "data", "node.csv")
        default_link = os.path.join(_URA_ROOT, sub, "data", "link.csv")
        if args.network == "bus":
            node_path = _c("BUS_NODE_CSV", None) or default_node
            link_path = _c("BUS_LINK_CSV", None) or default_link
        else:
            node_path = _c("ION_NODE_CSV", None) or default_node
            link_path = _c("ION_LINK_CSV", None) or default_link
        if not os.path.isfile(node_path) or not os.path.isfile(link_path):
            hint = (
                "build_bus_network"
                if args.network == "bus"
                else "build_ion_network"
            )
            parser.error(
                f"Missing {node_path} or {link_path} — run {hint} first."
            )
        nodes = pd.read_csv(node_path)
        links = pd.read_csv(link_path)
        if "zone_id" not in nodes.columns:
            parser.error(f"{node_path} has no zone_id column — rebuild with TAZ shapefile / assign_zone_id=True.")

        r = shortest_path_transit_zone_to_zone(
            nodes,
            links,
            args.orig,
            args.dest,
            cost=args.cost,
            verbose=True,
            list_all_candidates=list_all,
        )
        print(
            "found:",
            r["found"],
            "via nodes",
            r.get("orig_node_id"),
            "->",
            r.get("dest_node_id"),
        )
    else:
        import geopandas as gpd

        default_gpkg = os.path.join(
            _URA_ROOT, "road_network", "arcgis_export", "road_network.gpkg"
        )
        gpkg = _c("ROAD_GPKG_PATH", None) or default_gpkg
        if not os.path.isfile(gpkg):
            parser.error(f"Missing {gpkg} — run road_network build with export_arcgis first.")
        nodes = gpd.read_file(gpkg, layer="nodes")
        edges = gpd.read_file(gpkg, layer="links")
        if "zone_id" not in nodes.columns:
            parser.error("Road nodes layer has no zone_id — rebuild roads with assign_zone_id=True.")

        r = shortest_path_road_zone_to_zone(
            nodes,
            edges,
            args.orig,
            args.dest,
            weight_field=args.weight_field,
            verbose=True,
            list_all_candidates=list_all,
        )
        print(
            "found:",
            r["found"],
            "via nodes",
            r.get("orig_node_id"),
            "->",
            r.get("dest_node_id"),
            "total_cost:",
            r.get("total_cost"),
        )
