"""
Binary logit split of total OD demand between **road (car)** and **ION (transit)** using
skims between the same eight TAZ centroids on each network.

**Default utility (binary logit, generalized cost)**

Built-in utilities match :mod:`road_network.config` and :mod:`transit.shortest_path` (same as
shortest-path GC):

* **Car:** ``GC = (distance_km × COST_PER_KM) + (travel_time_h × VALUE_OF_TIME)`` using road skims
  of **time + distance** on **minimum-time** paths.
* **ION:** ``GC = FARE + (WAITING_TIME + T_ivt_min) × VOT_per_min`` (skim time = in-vehicle).

Systematic utility is ``V_m = ASC_m − GC_m`` (**ASC** in **dollars**, same units as GC).

**logit_scale** multiplies all utilities before ``exp`` (sharper vs flatter choice probabilities).

**Custom utilities** — Pass either ``utilities_odm`` (array ``(Z,Z,2)``) or
``utility_fn(t_car, t_ion) -> (Z,Z,2)`` to override GC; ``asc_*`` are ignored in that case.

Road paths are **not** minimum-generalized-cost unless the AequilibraE graph uses GC as the cost field.

Workflow
--------
1. Load total trips from ``od_matrix.csv`` (TAZ×TAZ long format).
2. Open each AequilibraE project, set centroids, skim **road: time + distance** (min-time paths)
   and **ION: time** (NetworkSkimming).
3. Reorder skims to **TAZ_SUBSET_EIGHT** order.
4. Build ``V_m = ASC_m − GC_m`` (or custom utilities) and apply :func:`allocate_total_demand_to_modes`.
5. Write two long-format CSVs for ``tap.py --layer road`` / ``--layer ion``.

**Congested** times (optional): ``--road-results`` / ``--ion-results`` from ``assig.results()``
(``link_id``, ``Congested_Time_Max``).

Example::

    python traffic_assignment_problem/mode_split.py --asc-car 0 --asc-ion 0

    # After TAP, congested skims for an iterative split::
    python traffic_assignment_problem/mode_split.py \\
        --road-results road_results.csv --ion-results ion_results.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable, Literal, Optional, Tuple

import numpy as np
import pandas as pd

_TAP_DIR = Path(__file__).resolve().parent
if str(_TAP_DIR) not in sys.path:
    sys.path.insert(0, str(_TAP_DIR))
_REPO_ROOT = _TAP_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from multi_layer_assignment import (
    DEFAULT_ION_NODE_CSV,
    DEFAULT_ROAD_NODE_CSV,
    MultiLayerPaths,
    TAZ_SUBSET_EIGHT,
    allocate_total_demand_to_modes,
    get_graph_for_mode,
    open_layer_project,
    pick_centroid_node_ids_per_zone_from_node_csv,
    reorder_skim_to_taz_order,
    set_centroids_by_node_ids,
    skim_graph_matrices,
    skim_travel_time_matrix,
    suggest_cost_and_capacity_fields,
    total_od_matrix_taz_order_from_long_csv,
    validate_aequilibrae_project_network,
)

from road_network.config import COST_PER_KM, VALUE_OF_TIME as ROAD_VOT_PER_HOUR
from transit.shortest_path import (
    DEFAULT_FARE,
    DEFAULT_WAITING_TIME,
    DEFAULT_VALUE_OF_TIME as TRANSIT_VOT_PER_MIN,
)

LayerName = Literal["road", "ion"]


def _distance_field_for_graph(graph) -> Optional[str]:
    for c in ("distance", "length_m", "length"):
        if c in graph.graph.columns:
            return c
    return None


def _apply_congested_times_from_results_csv(graph, results_path: Path, time_field: str) -> None:
    """Replace ``time_field`` on ``graph.graph`` with ``Congested_Time_Max`` from assignment CSV."""
    df = pd.read_csv(results_path)
    if "link_id" not in df.columns:
        raise ValueError(f"{results_path} must have link_id column")
    if "Congested_Time_Max" not in df.columns:
        raise ValueError(f"{results_path} must have Congested_Time_Max (from tap assig.results())")
    df = df.set_index("link_id")
    gdf = graph.graph.copy()
    lid = gdf["link_id"].to_numpy()
    ct = df["Congested_Time_Max"].reindex(lid).to_numpy()
    base = gdf[time_field].to_numpy(dtype=np.float64)
    use = np.where(np.isfinite(ct) & (ct > 0), ct, base)
    gdf[time_field] = use
    graph.graph = gdf
    graph.set_graph(time_field)


def run_mode_split(
    *,
    od_csv: Path,
    asc_car: float = 0.0,
    asc_ion: float = 0.0,
    logit_scale: float = 1.0,
    block_centroid_flows: bool = False,
    road_results_csv: Optional[Path] = None,
    ion_results_csv: Optional[Path] = None,
    out_car: Optional[Path] = None,
    out_ion: Optional[Path] = None,
    verbose: bool = True,
    utilities_odm: Optional[np.ndarray] = None,
    utility_fn: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Parameters
    ----------
    utilities_odm :
        If given, shape ``(Z, Z, 2)`` with ``[...,0]`` = car utility, ``[...,1]`` = ION utility
        (TAZ order). Skips generalized cost; ``asc_*`` are ignored.
    utility_fn :
        ``fn(t_car, t_ion) -> ndarray`` with shape ``(Z, Z, 2)``, same semantics. Skim arrays are
        in TAZ order; use ``np.inf`` / ``nan`` for unavailable OD if your logit should exclude them.
        Skims only **time** (no road distance); use ``utilities_odm`` if you need full control.

    Returns
    -------
    total_od, od_car, od_ion, utilities
        All shape (Z, Z) in **TAZ_SUBSET_EIGHT** order; ``utilities`` is (Z, Z, 2) for [car, ion].
    """
    paths = MultiLayerPaths()
    total_od = total_od_matrix_taz_order_from_long_csv(od_csv, TAZ_SUBSET_EIGHT)

    need_road_distance = utilities_odm is None and utility_fn is None

    def _layer_skm(
        layer: LayerName,
        mode: str,
        node_csv: Path,
        results_csv: Optional[Path],
        *,
        road_need_distance: bool,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        proj_path = paths.road_project if layer == "road" else paths.ion_project
        project = open_layer_project(proj_path)
        try:
            validate_aequilibrae_project_network(project)
            pick, _ = pick_centroid_node_ids_per_zone_from_node_csv(node_csv, TAZ_SUBSET_EIGHT)
            set_centroids_by_node_ids(project, pick)
            graph = get_graph_for_mode(project, mode)
            tf, _ = suggest_cost_and_capacity_fields(graph)
            if not tf:
                raise RuntimeError(f"No time field on {layer} graph")
            if results_csv is not None:
                _apply_congested_times_from_results_csv(graph, Path(results_csv), tf)
            pick_arr = np.array(pick, dtype=np.int64)
            if layer == "road" and road_need_distance:
                df = _distance_field_for_graph(graph)
                if df is None:
                    raise RuntimeError(
                        "Road graph has no `distance`, `length_m`, or `length` column; "
                        "cannot compute distance × COST_PER_KM for generalized cost."
                    )
                skim_raw, gc = skim_graph_matrices(
                    graph,
                    [tf, df],
                    cost_field=tf,
                    block_centroid_flows=block_centroid_flows,
                )
                t_re = reorder_skim_to_taz_order(skim_raw[:, :, 0], gc, pick_arr)
                d_re = reorder_skim_to_taz_order(skim_raw[:, :, 1], gc, pick_arr)
                return t_re, d_re
            skim_raw, gc = skim_travel_time_matrix(
                graph,
                time_field=tf,
                block_centroid_flows=block_centroid_flows,
            )
            return reorder_skim_to_taz_order(skim_raw, gc, pick_arr), None
        finally:
            try:
                project.close()
            except Exception:
                pass

    if verbose:
        if need_road_distance:
            print("Skimming road (car) network… (time + distance on min-time paths)")
        else:
            print("Skimming road (car) network…")
    t_car, d_road_m = _layer_skm(
        "road", "c", DEFAULT_ROAD_NODE_CSV, road_results_csv, road_need_distance=need_road_distance
    )
    if verbose:
        print("Skimming ION (transit) network…")
    t_ion, _ = _layer_skm(
        "ion", "t", DEFAULT_ION_NODE_CSV, ion_results_csv, road_need_distance=False
    )

    if utilities_odm is not None and utility_fn is not None:
        raise ValueError("Pass at most one of utilities_odm or utility_fn.")
    if utilities_odm is not None:
        utilities = np.asarray(utilities_odm, dtype=np.float64)
        if utilities.ndim != 3 or utilities.shape[2] != 2:
            raise ValueError(f"utilities_odm must have shape (Z, Z, 2); got {utilities.shape}")
        if utilities.shape[:2] != total_od.shape:
            raise ValueError("utilities_odm leading dimensions must match total_od")
    elif utility_fn is not None:
        utilities = np.asarray(utility_fn(t_car, t_ion), dtype=np.float64)
        if utilities.ndim != 3 or utilities.shape[2] != 2:
            raise ValueError(f"utility_fn must return shape (Z, Z, 2); got {utilities.shape}")
    else:
        if d_road_m is None:
            raise RuntimeError("internal: expected road distance skims for generalized cost")
        gc_car = (d_road_m / 1000.0) * COST_PER_KM + (t_car / 60.0) * ROAD_VOT_PER_HOUR
        gc_ion = DEFAULT_FARE + (DEFAULT_WAITING_TIME + t_ion) * TRANSIT_VOT_PER_MIN
        gc_car = np.where(np.isfinite(gc_car), gc_car, np.inf)
        gc_ion = np.where(np.isfinite(gc_ion), gc_ion, np.inf)
        U_car = asc_car - gc_car
        U_ion = asc_ion - gc_ion
        utilities = np.stack([U_car, U_ion], axis=-1)

    mode_od = allocate_total_demand_to_modes(total_od, utilities, scale=logit_scale)
    od_car = mode_od[:, :, 0]
    od_ion = mode_od[:, :, 1]

    out_car = out_car or (_TAP_DIR / "od_matrix_car.csv")
    out_ion = out_ion or (_TAP_DIR / "od_matrix_ion.csv")
    _write_long_od(out_car, od_car, TAZ_SUBSET_EIGHT)
    _write_long_od(out_ion, od_ion, TAZ_SUBSET_EIGHT)

    if verbose:
        print(f"Wrote {out_car.resolve()} and {out_ion.resolve()}")
        print(
            f"  Total trips in: {total_od.sum():.4f}; car assigned: {od_car.sum():.4f}; "
            f"ION assigned: {od_ion.sum():.4f}"
        )
        if utilities_odm is not None or utility_fn is not None:
            print("  Utilities: custom (utilities_odm or utility_fn)")
        else:
            print(
                f"  Generalized cost: road COST_PER_KM={COST_PER_KM}, VOT $/h={ROAD_VOT_PER_HOUR}; "
                f"ION fare={DEFAULT_FARE}, wait={DEFAULT_WAITING_TIME} min, VOT $/min={TRANSIT_VOT_PER_MIN}; "
                f"ASC_car={asc_car}, ASC_ion={asc_ion}, logit_scale={logit_scale}"
            )

    return total_od, od_car, od_ion, utilities


def _write_long_od(path: Path, mat: np.ndarray, taz_order: Tuple[int, ...]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, o in enumerate(taz_order):
        for j, d in enumerate(taz_order):
            rows.append({"zone_id_from": int(o), "zone_id_to": int(d), "demand": float(mat[i, j])})
    pd.DataFrame(rows).to_csv(path, index=False)


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Logit split of OD between car (road) and ION (transit) using dollar generalized cost."
    )
    p.add_argument(
        "--od-matrix",
        type=Path,
        default=_TAP_DIR / "od_matrix.csv",
        help="Long-format total demand (TAZ columns).",
    )
    p.add_argument(
        "--asc-car",
        type=float,
        default=0.0,
        help="Alternative-specific constant for car ($), subtracted from dollar GC.",
    )
    p.add_argument(
        "--asc-ion",
        type=float,
        default=0.0,
        help="Alternative-specific constant for ION ($), subtracted from dollar GC.",
    )
    p.add_argument("--logit-scale", type=float, default=1.0, help="Scale on utilities before softmax.")
    p.add_argument(
        "--block-centroid-flows",
        action="store_true",
        help="Skim with blocked centroid flows (match TAP if you use this there).",
    )
    p.add_argument(
        "--road-results",
        type=Path,
        default=None,
        help="Optional CSV from road assig.results() with link_id, Congested_Time_Max.",
    )
    p.add_argument(
        "--ion-results",
        type=Path,
        default=None,
        help="Optional CSV from ion assig.results() with link_id, Congested_Time_Max.",
    )
    p.add_argument("--out-car", type=Path, default=None, help="Output CSV for car OD (default od_matrix_car.csv).")
    p.add_argument("--out-ion", type=Path, default=None, help="Output CSV for ION OD (default od_matrix_ion.csv).")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = _parse_args(argv)
    run_mode_split(
        od_csv=args.od_matrix,
        asc_car=args.asc_car,
        asc_ion=args.asc_ion,
        logit_scale=args.logit_scale,
        block_centroid_flows=args.block_centroid_flows,
        road_results_csv=args.road_results,
        ion_results_csv=args.ion_results,
        out_car=args.out_car,
        out_ion=args.out_ion,
        verbose=True,
    )


if __name__ == "__main__":
    main()
