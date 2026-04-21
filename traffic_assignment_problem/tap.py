"""
Static traffic assignment (TAP) for the eight-TAZ subset on the **road** network.

Uses :mod:`road_assignment` to pick centroid nodes per TAZ, set centroids on the
AequilibraE project, load trips from ``od_matrix.csv`` (TAZ→TAZ long format), and run
``TrafficAssignment``.

**Centroid blocking:** By default we set ``block_centroid_flows=False``. Your centroids are
**real road nodes** (intersections with ``zone_id``); with blocking on, almost no OD pair has
a path that avoids passing through another centroid, so assignment stays all zeros. Use
``--block-centroid-flows`` only if you use separate connector-style centroids.

**Road capacity:** ``tap.py`` does not set capacities itself. They must exist on links in
the AequilibraE project (from GMNS). Rebuild the road network + project after editing
``road_network.config.CAPACITY_VPHPL_BY_CARTO_CLASS`` so ``link.csv`` includes a
``capacity`` column (veh/h per direction ≈ vphpl × lanes).

Run from the **URA repo root**::

    python traffic_assignment_problem/tap.py

Default graph mode is **c** (car) — use ``--mode`` to override.

After each run, results are written to ``traffic_assignment_problem/arcgis_export/tap_assignment_road.gpkg``
unless you pass ``--no-export-gpkg``. Override the path with ``--export-gpkg PATH``.
The default ``assignment_links`` layer includes **all** links (zeros on unused links). Use
``--export-gpkg-flow-only`` to write only links with ``demand_tot > 0``, or filter in ArcGIS
(``demand_tot > 0``).

Optional sample shortest-path layers::

    python traffic_assignment_problem/tap.py --export-path 811:836 --export-path 785:797

List valid ``--mode`` values for the road project::

    python traffic_assignment_problem/tap.py --list-modes
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# ``python traffic_assignment_problem/tap.py`` from repo root
_TAP_DIR = Path(__file__).resolve().parent
if str(_TAP_DIR) not in sys.path:
    sys.path.insert(0, str(_TAP_DIR))

from road_assignment import (
    DEFAULT_ROAD_NODE_CSV,
    RoadNetworkPaths,
    TAZ_SUBSET_EIGHT,
    build_traffic_assignment,
    create_memory_od_matrix,
    fill_od_from_long_table,
    get_graph_for_mode,
    open_layer_project,
    pick_centroid_node_ids_per_zone_from_node_csv,
    set_centroids_by_node_ids,
)

from assignment_export import export_assignment_gpkg

# Long-format OD: TAZ → TAZ trips. Columns: zone_id_from, zone_id_to, demand
OD_MATRIX_CSV = _TAP_DIR / "od_matrix.csv"


def _centroids_without_outgoing_links(graph, centroid_node_ids):
    """Centroid node IDs that have no outgoing links in the graph (AoN cannot load trips from them)."""
    bad = []
    for c in centroid_node_ids:
        c = int(c)
        i = int(graph.nodes_to_indices[c])
        if graph.fs[i] == graph.fs[i + 1]:
            bad.append(c)
    return bad


def load_od_matrix_into_demand(
    demand,
    od_csv: Path,
    *,
    centroid_pick: list,
    core_name: str = "demand",
) -> None:
    """
    Fill ``demand`` from ``od_csv`` using TAZ columns (mapped to centroid node IDs).

    CSV must have columns ``zone_id_from``, ``zone_id_to``, ``demand`` (extra spaces
    around names are stripped). ``centroid_pick`` is one node per TAZ in
    ``TAZ_SUBSET_EIGHT`` order (from ``pick_centroid_node_ids_per_zone_from_node_csv``).
    """
    if not od_csv.is_file():
        raise FileNotFoundError(f"OD matrix CSV not found: {od_csv}")

    od_df = pd.read_csv(od_csv)
    od_df.columns = [str(c).strip() for c in od_df.columns]
    need = {"zone_id_from", "zone_id_to", "demand"}
    if not need.issubset(od_df.columns):
        raise ValueError(
            f"{od_csv} needs columns {sorted(need)}; found {list(od_df.columns)}"
        )

    zone_to_node = {int(z): int(n) for z, n in zip(TAZ_SUBSET_EIGHT, centroid_pick)}
    od_df = od_df.copy()
    od_df["zone_id_from"] = od_df["zone_id_from"].astype(int)
    od_df["zone_id_to"] = od_df["zone_id_to"].astype(int)
    od_df["demand"] = pd.to_numeric(od_df["demand"], errors="coerce").fillna(0.0)
    od_df["o_node"] = od_df["zone_id_from"].map(zone_to_node)
    od_df["d_node"] = od_df["zone_id_to"].map(zone_to_node)
    bad = od_df["o_node"].isna() | od_df["d_node"].isna()
    if bad.any():
        rows = od_df.loc[bad, ["zone_id_from", "zone_id_to"]]
        raise ValueError(
            "Some TAZ codes in od_matrix.csv are not in TAZ_SUBSET_EIGHT:\n"
            + rows.to_string(index=False)
        )

    fill_od_from_long_table(
        demand,
        od_df,
        "o_node",
        "d_node",
        "demand",
        core_name=core_name,
    )


def _results_dataframe_with_link_id_column(results: pd.DataFrame) -> pd.DataFrame:
    """
    AequilibraE ``assig.results()`` often stores ``link_id`` on the **index**, not as a column.
    Match :func:`assignment_export.merge_assignment_results_to_links` behavior.
    """
    res = results.copy()
    if "link_id" in res.columns:
        return res
    res = res.reset_index()
    if "link_id" in res.columns:
        return res
    if "index" in res.columns:
        return res.rename(columns={"index": "link_id"})
    if "level_0" in res.columns:
        return res.rename(columns={"level_0": "link_id"})
    for alt in ("id", "link", "link_idx"):
        if alt in res.columns:
            return res.rename(columns={alt: "link_id"})
    raise ValueError(
        "Could not recover link_id from assignment results (expected index or link_id column). "
        f"Columns after reset_index: {list(res.columns)}"
    )


def export_assignment_link_times_csv(results: pd.DataFrame, path: Path) -> None:
    """Write ``link_id`` and ``Congested_Time_Max`` for downstream analysis."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    res = _results_dataframe_with_link_id_column(results)
    if "Congested_Time_Max" not in res.columns:
        raise ValueError(
            f"Assignment results missing Congested_Time_Max (have {list(res.columns)}); "
            "cannot export link times CSV."
        )
    out = res[["link_id", "Congested_Time_Max"]].copy()
    out["link_id"] = out["link_id"].astype(int)
    out.to_csv(path, index=False)
    print(f"\nLink results CSV written: {path.resolve()}")


def list_graph_modes(project) -> None:
    graphs = project.network.graphs
    if not graphs:
        print("No graphs: run project.network.build_graphs() on the project first.")
        return
    print("Available modes:", ", ".join(repr(m) for m in sorted(graphs.keys())))


def run_tap(
    mode: str = "c",
    *,
    od_matrix_path: Optional[Path] = None,
    max_iter: int = 50,
    rgap_target: float = 1e-4,
    node_csv: Optional[Path] = None,
    algorithm: Optional[str] = None,
    class_name: Optional[str] = None,
    block_centroid_flows: bool = False,
    export_gpkg: Optional[Path] = None,
    export_paths: Optional[Sequence[Tuple[int, int]]] = None,
    export_results_csv: Optional[Path] = None,
    export_gpkg_flow_only: bool = False,
):
    """
    Configure centroids for :data:`TAZ_SUBSET_EIGHT`, load trips from ``od_matrix.csv``,
    and run TAP on ``mode``.
    """
    paths = RoadNetworkPaths()
    project = open_layer_project(paths.road_project)

    csv = Path(node_csv) if node_csv is not None else DEFAULT_ROAD_NODE_CSV
    centroid_pick, summary = pick_centroid_node_ids_per_zone_from_node_csv(
        csv, TAZ_SUBSET_EIGHT
    )
    print(f"TAZ subset (order): {TAZ_SUBSET_EIGHT}")
    print("Representative node per TAZ (from GMNS node.csv):")
    print(summary.to_string(index=False))

    set_centroids_by_node_ids(project, centroid_pick)
    graph = get_graph_for_mode(project, mode)

    centroid_ids = list(graph.centroids)
    print(f"Graph mode={mode!r}: {len(centroid_ids)} centroids (DB order): {centroid_ids}")
    print(f"  Links in this graph: {graph.num_links}")

    demand = create_memory_od_matrix(centroid_ids, core_name="demand")
    od_path = Path(od_matrix_path) if od_matrix_path is not None else OD_MATRIX_CSV
    print(f"Loading OD demand from: {od_path.resolve()}")
    load_od_matrix_into_demand(demand, od_path, centroid_pick=list(centroid_pick))
    demand.computational_view(["demand"])
    demand.matrix_view = np.asarray(demand.matrix_view, dtype=np.float64, order="C")
    print(f"Total trips in matrix: {float(np.asarray(demand.matrix_view).sum()):.4f}")
    if not np.array_equal(np.asarray(demand.index, dtype=np.uint64), np.asarray(graph.centroids, dtype=np.uint64)):
        raise ValueError(
            "OD matrix zone index order does not match graph.centroids — check create_memory_od_matrix / fill_od."
        )

    dead = _centroids_without_outgoing_links(graph, centroid_ids)
    if dead:
        print(
            "WARNING: These centroids have no outgoing links (cannot start trips): "
            f"{dead}. Check graph mode {mode!r} on incident links (link ``modes`` in GMNS) and connectivity."
        )

    algo = algorithm if algorithm is not None else "bfw"
    cls_name = class_name if class_name is not None else f"road_{mode}"

    assig = build_traffic_assignment(
        project,
        graph,
        demand,
        class_name=cls_name,
        algorithm=algo,
        block_centroid_flows=block_centroid_flows,
    )
    assig.max_iter = max_iter
    assig.rgap_target = float(rgap_target)

    print(
        f"Running TAP (algorithm={algo!r}, class={cls_name!r}, "
        f"block_centroid_flows={graph.block_centroid_flows})..."
    )
    assig.execute()
    print("Assignment complete.")

    try:
        print("\nConvergence report:")
        print(assig.report().to_string(index=False))
    except Exception:
        pass

    results = assig.results()
    if "demand_tot" in results.columns:
        s = float(results["demand_tot"].sum())
        npos = int((results["demand_tot"] > 1e-9).sum())
        print(f"\nLink demand_tot: sum={s:.4f} across {len(results)} links; {npos} links with flow > 0")
        if s <= 0 and float(np.asarray(demand.matrix_view).sum()) > 0:
            print(
                "Matrix has trips but no link volumes: paths may be missing (disconnected network / wrong mode), "
                "or centroids have no outgoing links (see WARNING above)."
            )
    print("\nAssignment results (head — low link_id rows are often unused):")
    print(results.head())
    if "demand_tot" in results.columns and (results["demand_tot"] > 0).any():
        print("\nTop 5 links by demand_tot:")
        print(
            results.nlargest(5, "demand_tot")[
                [c for c in ("demand_ab", "demand_ba", "demand_tot") if c in results.columns]
            ].to_string()
        )

    if export_results_csv is not None:
        try:
            export_assignment_link_times_csv(results, Path(export_results_csv))
        except Exception as e:
            print(f"\nWarning: --export-results-csv failed: {e}")

    if export_gpkg is not None:
        out = Path(export_gpkg)
        pairs = list(export_paths) if export_paths else []
        try:
            tf = assig.time_field
            export_assignment_gpkg(
                results,
                out,
                graph=graph,
                time_field=tf,
                path_od_pairs=pairs if pairs else None,
                assignment_links_flow_only=export_gpkg_flow_only,
            )
            print(f"\nGeoPackage written: {out.resolve()}")
            print("  Layers: assignment_links (link volumes + congested times);")
            if pairs:
                print("            path_links, path_nodes (shortest paths on post-assignment costs per --export-path).")
        except Exception as e:
            print(f"\nWarning: GeoPackage export failed: {e}")

    try:
        project.close()
    except Exception:
        pass

    return assig, graph, results


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Road network TAP for eight TAZs.")
    p.add_argument(
        "--mode",
        default="c",
        help="AequilibraE graph mode (default: c for car).",
    )
    p.add_argument(
        "--list-modes",
        action="store_true",
        help="Print graph modes for the road project and exit.",
    )
    p.add_argument(
        "--od-matrix",
        type=Path,
        default=None,
        help=f"Long-format OD CSV (default: {OD_MATRIX_CSV.name} next to tap.py).",
    )
    p.add_argument("--max-iter", type=int, default=50)
    p.add_argument("--rgap", type=float, default=1e-4)
    p.add_argument(
        "--algorithm",
        default=None,
        help="Override assignment algorithm (e.g. bfw, msa, all-or-nothing).",
    )
    p.add_argument(
        "--node-csv",
        type=Path,
        default=None,
        help="Optional GMNS node.csv with zone_id (default: road_network/data/gmns/node.csv).",
    )
    p.add_argument(
        "--class-name",
        default=None,
        help="TrafficClass name (default: road_<mode>).",
    )
    p.add_argument(
        "--block-centroid-flows",
        action="store_true",
        help="Forbid paths through other centroids (classic four-step). Off by default — "
        "needed off when centroids are real road intersections.",
    )
    p.add_argument(
        "--no-export-gpkg",
        action="store_true",
        help="Skip writing traffic_assignment_problem/arcgis_export/tap_assignment_road.gpkg after assignment.",
    )
    p.add_argument(
        "--export-gpkg",
        nargs="?",
        const="__default__",
        default=None,
        metavar="PATH",
        help="GeoPackage output path (default without this flag: tap_assignment_road.gpkg under arcgis_export/). "
        "Pass this flag alone to force that same default path explicitly.",
    )
    p.add_argument(
        "--export-path",
        action="append",
        metavar="O:D",
        default=None,
        help="Repeatable. Centroid node pair for path_links/path_nodes layers (e.g. 811:836).",
    )
    p.add_argument(
        "--export-gpkg-flow-only",
        action="store_true",
        help="GeoPackage assignment_links: keep only links with demand_tot > 0 (default: all network links).",
    )
    p.add_argument(
        "--export-results-csv",
        nargs="?",
        const="__default__",
        default=None,
        metavar="PATH",
        help=(
            "After assignment, write link_id and Congested_Time_Max to CSV. "
            "Default path: tap_assignment_road_link_results.csv next to tap.py."
        ),
    )
    return p.parse_args(argv)


def _parse_od_pair(s: str) -> Tuple[int, int]:
    s = str(s).strip()
    for sep in (":", ","):
        if sep in s:
            a, b = s.split(sep, 1)
            return int(a.strip()), int(b.strip())
    raise ValueError(f"Expected O:D or O,D, got {s!r}")


def main(argv: Optional[list[str]] = None) -> None:
    args = _parse_args(argv)

    if args.list_modes:
        paths = RoadNetworkPaths()
        project = open_layer_project(paths.road_project)
        try:
            list_graph_modes(project)
        finally:
            try:
                project.close()
            except Exception:
                pass
        return

    default_gpkg = _TAP_DIR / "arcgis_export" / "tap_assignment_road.gpkg"
    export_gpkg: Optional[Path] = None
    if args.no_export_gpkg:
        export_gpkg = None
    elif args.export_gpkg is not None:
        export_gpkg = default_gpkg if args.export_gpkg == "__default__" else Path(args.export_gpkg)
    else:
        export_gpkg = default_gpkg

    path_pairs: Optional[List[Tuple[int, int]]] = None
    if args.export_path:
        path_pairs = [_parse_od_pair(x) for x in args.export_path]
        if args.no_export_gpkg:
            print("Warning: --export-path ignored because --no-export-gpkg was set.")

    export_results_csv: Optional[Path] = None
    if args.export_results_csv is not None:
        export_results_csv = (
            _TAP_DIR / "tap_assignment_road_link_results.csv"
            if args.export_results_csv == "__default__"
            else Path(args.export_results_csv)
        )

    run_tap(
        args.mode,
        od_matrix_path=args.od_matrix,
        max_iter=args.max_iter,
        rgap_target=args.rgap,
        node_csv=args.node_csv,
        algorithm=args.algorithm,
        class_name=args.class_name,
        block_centroid_flows=args.block_centroid_flows,
        export_gpkg=export_gpkg,
        export_paths=path_pairs,
        export_results_csv=export_results_csv,
        export_gpkg_flow_only=args.export_gpkg_flow_only,
    )


if __name__ == "__main__":
    main()
