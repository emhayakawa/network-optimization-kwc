"""
Setup for static traffic assignment across three network layers (road, bus, ION).

Can I use TAZs, a small zone subset (e.g. 8 TAZs), and one OD matrix, and have road
congestion choose road vs bus vs ION in **one** AequilibraE run?
------------------------------------------------------------------------
**Partially.**

* **TAZ-based OD** and **extracting an 8×8 (or 8-zone) submatrix** — yes. Use the same
  TAZ IDs as row/column labels, then ``extract_taz_submatrix`` (below).
* **Mapping TAZ → centroid node** — must be done **separately per layer** (road graph
  node IDs ≠ bus node IDs ≠ ION node IDs). Typically you pick one centroid/connector
  node per TAZ on each network (or build connectors). ``create_memory_od_matrix`` then
  uses that layer's centroid **node_id** list in the same order as your TAZ list.
* **Simultaneous equilibrium** (one solver, one graph, congestion and mode choice fully
  joint) — **not** what you get from three independent projects. AequilibraE
  ``TrafficAssignment`` equilibrates **one** link graph (optionally **multiple user
  classes on that same graph**). It does not automatically couple your road links to
  your bus line graph and ION graph.

**What does implement “road congestion ↔ mode choice” in practice?**

1. **Iterative combined model (classic four-step extension)** — repeated until stable:

   * Skim **generalized cost** (or time + money) on **road**, **bus**, and **ION**
     between the same TAZ centroids (AequilibraE skimming, or shortest paths).
   * **Mode split** (e.g. multinomial logit) using those skims + ASCs/fares;
     ``logit_mode_shares`` / ``allocate_total_demand_to_modes`` below split total
     person trips into car / bus / ION matrices.
   * **Assign** car on the **road** network with BPR (congestion updates skim times).
   * **Assign** bus / ION on their graphs (often AON or fixed headway/capacity unless
     you model bus-in-traffic explicitly).
   * Loop: update road skims from congested times, optionally update bus times if your
     model ties them to traffic, re-split modes, re-assign.

   Equilibrium here is **fixed-point** across iterations, not a single closed form.

2. **Supernetwork / hyperpath** — one augmented graph encoding transfers and modes;
   much more setup, still often combined with logit for mode/transfer choices.

3. **Agent-based (e.g. MATSim)** — agents choose modes and routes with day-to-day
   learning; strong fit for “who drives vs takes transit” but different workflow.

This repo already builds separate AequilibraE projects under ``road_network/``,
``bus_network/``, and ``ion_network/``. Each layer is a **different graph** with its
own node numbering. Assignment is therefore **per layer** unless you **merge** into one
project (see below).

One consolidated ``Project``, multiple ``modes`` (road + ION)
-------------------------------------------------------------
**If you want “one TAP” inside AequilibraE, that is one SQLite network** (one
``Project``). ``project.network.build_graphs()`` then builds **one directed graph per
mode** (e.g. ``'c'`` = car, ``'t'`` = transit). Each **link** row lists which modes may
use it; car-only and transit-only links live in the same ``links`` table.

* **ION already includes bus routes** in this repo — for a consolidated build you only
  need **road + ION** transit links, not a third “bus-only” copy (avoids duplicating the
  bus subgraph).
* **Merging** road GMNS and ION GMNS is **not automatic**: you must give every node and
  link a **globally unique** ID across both sources, set ``modes`` on each link, import
  once (or append carefully), and usually add **connector / transfer** links if paths
  should move between driving and transit inside the same assignment. That merged
  network is substantial custom GIS/network prep; this file does not build it.

**Do you need zone centroids?** **Yes**, for the usual static TAP formulation in
AequilibraE.

* Centroids are **nodes** flagged ``is_centroid = 1`` (or GMNS ``node_type=centroid``).
  They are the **only** places where OD demand is injected and attracted.
* Your **OD matrix** external indices must be those nodes’ **``node_id``** values, in the
  **same order** as ``graph.centroids`` (see ``create_memory_od_matrix``). TAZ codes
  (e.g. 22001) are **not** the matrix key unless you deliberately use ``node_id == TAZ``
  or you map TAZ → ``node_id`` yourself.
* **How TAP uses them:** for each OD pair ``(O, D)``, flow is loaded onto paths from
  centroid ``O`` to centroid ``D``. With ``set_blocked_centroid_flows(True)`` (default in
  ``build_traffic_assignment``), traffic **may not pass through other centroids** as
  intermediate nodes — a standard way to stop zones from acting as shortcuts.

**Eight-zone study set:** :data:`TAZ_SUBSET_EIGHT` lists the TAZ IDs used for a small
OD block. Use :func:`pick_centroid_node_ids_per_zone_from_node_csv` to pick one
representative **node_id** per TAZ on ION (or bus) GMNS exports, then
:func:`set_centroids_by_node_ids` on that layer’s project (repeat for road with a road
node table that has ``zone_id``). Use :func:`extract_assignment_submatrix_eight_zone` to
slice a full TAZ matrix down to those eight zones.

Why AequilibraE fits
--------------------
* **Road**: Full static user equilibrium (Frank–Wolfe family, BPR VDF) via
  ``TrafficAssignment`` + ``TrafficClass`` — the primary intended use case.
* **Bus / ION (line graphs)**: The GMNS graphs are transit **topology** graphs. You can
  run the same ``TrafficAssignment`` machinery with **all-or-nothing** or **MSA** and
  link capacities chosen to approximate line capacity / headway, or treat results as
  **loading on scheduled minimum-time paths** without congestion feedback. For true
  frequency-based transit assignment, AequilibraE also exposes ``TransitAssignment``
  (requires transit line tables / graph setup beyond this file).

Other methods (when not to use AequilibraE alone)
-------------------------------------------------
* **MATSim** — agent-based multimodal simulation; strong mode choice and congestion
  feedback; heavier runtime and data needs.
* **SUMO** — microscopic simulation; good for dynamic network loading and detailed transit.
* **Commercial suites (Emme, Visum, Cube)** — integrated multi-modal equilibrium and
  standard transit assignment procedures.
* **DTALite / open DTA tools** — dynamic traffic assignment focused on roads.
* **Split demand + assign** — estimate mode shares with a logit or rule-based splitter,
  then assign **car** on the road network and **transit** on bus/ION graphs separately.
  This is often the pragmatic alternative to a single joint equilibrium on one graph.

Centroids and matrices (required for assignment)
-----------------------------------------------
AequilibraE builds ``graph.centroids`` from nodes with ``is_centroid = 1`` (GMNS
``node_type == "centroid"`` on import). The ``TrafficClass`` constructor requires
``demand.index`` to match ``graph.centroids`` **exactly** (same IDs, same order).

If your builds have no centroids yet, mark connector or representative nodes (e.g. one
node per TAZ) before calling ``build_graphs()``. Then build an ``AequilibraeMatrix``
with ``create_memory_od_matrix`` below.

Example (road layer only, after centroids and demand exist)::

    from multi_layer_assignment import (
        MultiLayerPaths,
        open_layer_project,
        get_graph_for_mode,
        create_memory_od_matrix,
        build_traffic_assignment,
    )

    paths = MultiLayerPaths()
    road = open_layer_project(paths.road_project)
    g = get_graph_for_mode(road, "c")
    demand = create_memory_od_matrix(list(g.centroids), core_name="car")
    demand.computational_view(["car"])
    # ... fill demand.matrix_view[i,j] ...

    assig = build_traffic_assignment(road, g, demand, class_name="car")
    assig.execute()
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

if TYPE_CHECKING:
    from aequilibrae.matrix import AequilibraeMatrix

import numpy as np
import pandas as pd

# Repo root (parent of ``traffic_assignment_problem/``).
_REPO_ROOT = Path(__file__).resolve().parent.parent

# Default eight-TAZ subset for small-network TAP / skims (row/col order for submatrices).
TAZ_SUBSET_EIGHT: Tuple[int, ...] = (
    22001,
    21903,
    21414,
    21412,
    21417,
    21411,
    21410,
    21418,
)

# GMNS ``node.csv`` from this repo’s builds (``zone_id`` = TAZ).
DEFAULT_BUS_NODE_CSV: Path = _REPO_ROOT / "bus_network" / "data" / "node.csv"
DEFAULT_ION_NODE_CSV: Path = _REPO_ROOT / "ion_network" / "data" / "node.csv"
DEFAULT_ROAD_NODE_CSV: Path = _REPO_ROOT / "road_network" / "data" / "gmns" / "node.csv"


@dataclass
class MultiLayerPaths:
    """Default locations of the three AequilibraE project folders in this repo."""

    road_project: Path = _REPO_ROOT / "road_network" / "aequilibrae_project"
    bus_project: Path = _REPO_ROOT / "bus_network" / "aequilibrae_project"
    ion_project: Path = _REPO_ROOT / "ion_network" / "aequilibrae_project"


def open_layer_project(project_dir: Union[str, Path]):
    """
    Open an existing AequilibraE project from disk.

    Build the project first with ``road_network.build_road_network`` (``create_aequilibrae=True``),
    ``bus_network.aequilibrae_network.create_aequilibrae_project``, or
    ``ion_network.aequilibrae_network.create_aequilibrae_project``.
    """
    from aequilibrae import Project

    project_dir = Path(project_dir)
    if not project_dir.is_dir():
        raise FileNotFoundError(f"AequilibraE project not found: {project_dir}")
    project = Project()
    project.open(str(project_dir))
    return project


def open_all_layers(paths: Optional[MultiLayerPaths] = None) -> dict:
    """Return ``{"road": proj, "bus": proj, "ion": proj}`` for default paths."""
    paths = paths or MultiLayerPaths()
    return {
        "road": open_layer_project(paths.road_project),
        "bus": open_layer_project(paths.bus_project),
        "ion": open_layer_project(paths.ion_project),
    }


def get_graph_for_mode(project, mode: str = "c"):
    """
    Return the built graph for ``mode`` (usually ``'c'`` for car after GMNS import).

    Raises if graphs were not built (call ``project.network.build_graphs()`` first).
    """
    graphs = project.network.graphs
    if not graphs:
        raise RuntimeError("No graphs on project. Run project.network.build_graphs() first.")
    if mode not in graphs:
        available = ", ".join(sorted(graphs.keys()))
        raise KeyError(f"No graph for mode {mode!r}. Available: {available}")
    g = graphs[mode]
    if g.centroids is None or len(g.centroids) == 0:
        raise ValueError(
            "Graph has no centroids. Mark nodes with is_centroid=1 (or GMNS node_type "
            "centroid), then rebuild graphs."
        )
    return g


def graph_centroids(project, mode: str = "c") -> np.ndarray:
    """Convenience: centroid node IDs for a mode (numpy uint32 array)."""
    return get_graph_for_mode(project, mode).centroids


def create_memory_od_matrix(
    centroid_node_ids: Sequence[int],
    core_name: str = "demand",
) -> AequilibraeMatrix:
    """
    Create an in-memory OD matrix aligned to ``centroid_node_ids`` order.

    The sequence must match ``graph.centroids`` from the layer you assign on.
    After filling, call ``mat.computational_view([core_name])`` before passing to
    ``TrafficClass``.
    """
    from aequilibrae.matrix import AequilibraeMatrix

    ids = np.asarray(list(centroid_node_ids), dtype=np.uint32)
    if ids.size == 0:
        raise ValueError("centroid_node_ids is empty")
    if np.unique(ids).size != ids.size:
        raise ValueError("centroid_node_ids must be unique")

    mat = AequilibraeMatrix()
    mat.create_empty(zones=ids.size, matrix_names=[core_name], memory_only=True)
    mat.indices[:, 0] = ids.astype(np.uint64)
    mat.set_index(mat.index_names[0])
    mat.computational_view([core_name])
    mat.matrix_view.fill(0.0)
    return mat


def fill_od_from_long_table(
    mat: AequilibraeMatrix,
    od: pd.DataFrame,
    orig_col: str,
    dest_col: str,
    flow_col: str,
    *,
    core_name: Optional[str] = None,
) -> None:
    """
    Fill a matrix from a long-format table (origin, destination, trips).

    Row indices are resolved using ``mat.index`` (external zone / node IDs).
    """
    if core_name is None:
        names = mat.view_names or mat.names
        core_name = names[0]
    mat.computational_view([core_name])
    idx = {int(z): i for i, z in enumerate(mat.index)}
    for _, row in od.iterrows():
        o = int(row[orig_col])
        d = int(row[dest_col])
        if o not in idx or d not in idx:
            continue
        mat.matrix_view[idx[o], idx[d]] = float(row[flow_col])


def _first_present_column(cols: Iterable[str], candidates: Tuple[str, ...]) -> Optional[str]:
    s = set(cols)
    for c in candidates:
        if c in s:
            return c
    return None


def suggest_cost_and_capacity_fields(graph) -> Tuple[Optional[str], Optional[str]]:
    """
    Pick reasonable default column names on ``graph.graph`` for time and capacity.

    Prefers GMNS-style columns from ``link.csv`` used to build the project:
    ``travel_time_min`` (minutes) and scalar ``capacity`` (veh/h per link direction),
    then falls back to AequilibraE defaults (``free_flow_time``, ``travel_time_*``,
    ``capacity_*``).

    Returns (time_field, capacity_field); either may be None if missing — you must
    set link attributes in the project DB before assignment.
    """
    cols = graph.graph.columns
    time_field = _first_present_column(
        cols,
        (
            "travel_time_min",
            "free_flow_time",
            "travel_time",
            "travel_time_ab",
            "travel_time_ba",
            "time",
        ),
    )
    cap_field = _first_present_column(
        cols,
        ("capacity", "capacity_ab", "capacity_ba"),
    )
    return time_field, cap_field


def build_traffic_assignment(
    project,
    graph,
    demand: AequilibraeMatrix,
    class_name: str,
    *,
    time_field: Optional[str] = None,
    capacity_field: Optional[str] = None,
    skim_fields: Optional[List[str]] = None,
    algorithm: str = "bfw",
    vdf: str = "BPR",
    vdf_parameters: Optional[Mapping[str, Union[float, str]]] = None,
    block_centroid_flows: bool = True,
):
    """
    Configure ``TrafficAssignment`` for one layer (one graph + one demand matrix).

    Does **not** call ``execute()`` — so you can adjust ``max_iter`` / ``rgap_target``
    first.

    Parameters ``time_field`` and ``capacity_field`` default to columns detected via
    ``suggest_cost_and_capacity_fields`` (preferring ``travel_time_min`` and ``capacity``
    from GMNS ``link.csv`` when present on the graph); BPR requires positive capacities
    on all links used in the graph.
    """
    from aequilibrae.paths import TrafficAssignment, TrafficClass

    if time_field is None or capacity_field is None:
        t_guess, c_guess = suggest_cost_and_capacity_fields(graph)
        time_field = time_field or t_guess
        capacity_field = capacity_field or c_guess
    if not time_field or not capacity_field:
        raise ValueError(
            "Could not infer time/capacity fields on the graph. "
            "Pass time_field and capacity_field explicitly after adding them to links."
        )

    if skim_fields is None:
        skim_fields = [time_field]
        if "distance" in graph.graph.columns:
            skim_fields.append("distance")

    graph.set_graph(time_field)
    graph.set_skimming(skim_fields)
    graph.set_blocked_centroid_flows(block_centroid_flows)

    if demand.matrix_view.dtype != graph.default_types("float"):
        raise TypeError("Demand matrix computational view must be float64 for TrafficClass.")

    tc = TrafficClass(class_name, graph, demand)
    assig = TrafficAssignment(project)
    assig.set_classes([tc])
    assig.set_vdf(vdf)
    pars = vdf_parameters if vdf_parameters is not None else {"alpha": 0.15, "beta": 4.0}
    assig.set_vdf_parameters(dict(pars))
    assig.set_capacity_field(capacity_field)
    assig.set_time_field(time_field)
    assig.set_algorithm(algorithm)
    return assig


def pick_centroid_node_ids_per_zone_from_node_csv(
    node_csv: Union[str, Path],
    taz_ids: Optional[Sequence[int]] = None,
    *,
    zone_col: str = "zone_id",
    node_col: str = "node_id",
    strategy: str = "min_node_id",
) -> Tuple[List[int], pd.DataFrame]:
    """
    Pick one ``node_id`` per TAZ from a GMNS-style ``node.csv`` (bus / ION export).

    Use this to decide **which** network nodes become AequilibraE **centroids** for your
    eight-zone (or other) subset. Road centroids should come from the **road** node
    table (e.g. GeoPackage / CSV) that has ``zone_id`` on intersection nodes — same idea,
    possibly different paths.

    Parameters
    ----------
    node_csv :
        Path to ``node.csv`` with at least ``zone_id`` and ``node_id``.
    taz_ids :
        TAZs in **output order**. Default: :data:`TAZ_SUBSET_EIGHT`.
    zone_col, node_col :
        Column names for TAZ and node identifier.
    strategy :
        ``"min_node_id"`` — smallest ``node_id`` in that TAZ (deterministic).

    Returns
    -------
    centroid_node_ids :
        Length ``len(taz_ids)``, same order as ``taz_ids``.
    summary :
        One row per TAZ with chosen ``centroid_node_id`` and row count in file.
    """
    if taz_ids is None:
        taz_ids = TAZ_SUBSET_EIGHT

    path = Path(node_csv)
    if not path.is_file():
        raise FileNotFoundError(f"node.csv not found: {path}")

    df = pd.read_csv(path)
    for col in (zone_col, node_col):
        if col not in df.columns:
            raise KeyError(f"Column {col!r} missing from {path}. Found: {df.columns.tolist()}")

    zt = df[zone_col].astype(int)
    picked: List[int] = []
    rows_out: List[dict] = []

    for z in taz_ids:
        z = int(z)
        mask = zt == z
        g = df.loc[mask, node_col]
        if g.empty:
            raise ValueError(f"No rows with {zone_col}={z} in {path}")
        if strategy == "min_node_id":
            nid = int(g.astype(int).min())
        else:
            raise ValueError(f"Unknown strategy: {strategy!r}")

        picked.append(nid)
        rows_out.append(
            {
                "zone_id": z,
                "centroid_node_id": nid,
                "nodes_in_taz_in_file": int(mask.sum()),
            }
        )

    return picked, pd.DataFrame(rows_out)


_build_graphs_patch_installed = False


def _coerce_links_endpoints_sqlite(project) -> None:
    """
    Normalize links.a_node / b_node as INTEGER in SQLite.

    If endpoints were stored or read as text, AequilibraE's build_graphs() can leave them
    out of ``select_dtypes(np.number)`` and then fail with KeyError 'a_node'.

    We intentionally do **not** CAST ``direction`` here (must stay in {-1, 0, 1} for the
    graph builder; a bad CAST could introduce NULLs).
    """
    conn = getattr(project, "conn", None) or getattr(project, "db_connection", None)
    if conn is None:
        return

    def _run(db):
        try:
            db.execute(
                "UPDATE links SET a_node = CAST(a_node AS INTEGER), "
                "b_node = CAST(b_node AS INTEGER)"
            )
        except Exception:
            return
        if hasattr(db, "commit"):
            db.commit()

    try:
        if hasattr(conn, "__enter__"):
            with conn as db:
                _run(db)
        else:
            _run(conn)
    except Exception:
        pass


def _install_aequilibrae_build_graphs_patch() -> None:
    """
    Patch Network.build_graphs so link_id / a_node / b_node / direction are never dropped.

    Upstream uses ``valid_fields = select_dtypes(np.number) + ['modes']``, which omits
    endpoint columns when pandas dtypes are object or nullable in a way that excludes
    them from ``np.number`` (seen on Python 3.13 + some SQLite reads).
    """
    global _build_graphs_patch_installed
    if _build_graphs_patch_installed:
        return
    try:
        from aequilibrae.project.network.network import Network
    except ImportError:
        return
    if getattr(Network.build_graphs, "_ura_endpoint_patch", False):
        _build_graphs_patch_installed = True
        return

    def _build_graphs_fixed(self, fields=None, modes=None, limit_to_area=None):
        from aequilibrae.paths import Graph

        with self.project.db_connection as conn:
            if fields is None:
                field_names = conn.execute("PRAGMA table_info(links);").fetchall()
                ignore_fields = ["ogc_fid", "geometry"]
                all_fields = [f[1] for f in field_names if f[1] not in ignore_fields]
            else:
                fields.extend(["link_id", "a_node", "b_node", "direction", "modes"])
                all_fields = list(set(fields))

            if modes is None:
                modes = conn.execute("select mode_id from modes;").fetchall()
                modes = [m[0] for m in modes]
            elif isinstance(modes, str):
                modes = [modes]

            if limit_to_area is not None:
                from aequilibrae.utils.spatialite_utils import load_spatialite_extension

                load_spatialite_extension(conn)
                spatial_add = """ WHERE links.rowid in (
                                        select rowid from SpatialIndex where f_table_name = 'links' and
                                       search_frame = GeomFromWKB(?, 4326))"""

            sql = f"select {','.join(all_fields)} from links"

            sql_centroids = "select node_id from nodes where is_centroid=1 order by node_id;"
            centroids = np.array([i[0] for i in conn.execute(sql_centroids).fetchall()], np.uint32)
            centroids = centroids if centroids.shape[0] else None

            with pd.option_context("future.no_silent_downcasting", True):
                if limit_to_area is None:
                    df = pd.read_sql(sql, conn).fillna(value=np.nan).infer_objects(False)
                else:
                    sql = sql + spatial_add
                    df = (
                        pd.read_sql_query(sql, conn, params=(limit_to_area.wkb,))
                        .fillna(value=np.nan)
                        .infer_objects(False)
                    )
                    centroids = centroids[np.isin(centroids, df.a_node) | np.isin(centroids, df.b_node)]

            valid_fields = list(df.select_dtypes(np.number).columns)
            for _c in ("link_id", "a_node", "b_node", "direction"):
                if _c in df.columns and _c not in valid_fields:
                    valid_fields.append(_c)
            if "modes" in df.columns and "modes" not in valid_fields:
                valid_fields.append("modes")

        from aequilibrae.context import get_logger

        lonlat = self.nodes.lonlat.set_index("node_id")
        data = df[valid_fields]
        if data.shape[0] == 0:
            raise RuntimeError(
                "AequilibraE links table is empty (0 rows). Recreate the project from GMNS "
                "(e.g. road_network.build_network(create_aequilibrae=True))."
            )
        for m in modes:
            net = pd.DataFrame(data, copy=True)
            net.loc[~net.modes.str.contains(m, na=False), "b_node"] = net.loc[
                ~net.modes.str.contains(m, na=False), "a_node"
            ]

            g = Graph()
            g.mode = m
            g.network = net
            g.prepare_graph(centroids)
            g.set_blocked_centroid_flows(True)
            if centroids is None:
                get_logger().warning("Your graph has no centroids")
            g.lonlat_index = lonlat.loc[g.all_nodes]
            self.graphs[m] = g

    _build_graphs_fixed._ura_endpoint_patch = True
    Network.build_graphs = _build_graphs_fixed
    _build_graphs_patch_installed = True


def extract_assignment_submatrix_eight_zone(
    demand: np.ndarray,
    zone_ids: Sequence[int],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract the 8×8 OD block for :data:`TAZ_SUBSET_EIGHT` from a full TAZ-square matrix.

    ``zone_ids`` is the full TAZ ordering of rows/columns of ``demand`` (same length as
    each dimension).
    """
    return extract_taz_submatrix(demand, zone_ids, TAZ_SUBSET_EIGHT)


def validate_aequilibrae_project_network(project) -> None:
    """
    Raise ``RuntimeError`` if the project has no links (common when import failed or the
    folder was only partially created).
    """
    with project.db_connection as conn:
        n_links = int(conn.execute("SELECT COUNT(*) FROM links").fetchone()[0])
    if n_links == 0:
        raise RuntimeError(
            "This AequilibraE project has 0 links in ``links``. It cannot run TAP or "
            "build_graphs. Recreate the project from GMNS (e.g. "
            "``road_network.build_network(create_aequilibrae=True)`` or "
            "``python ion_network/create_aequilibrae_project.py``)."
        )


def set_centroids_by_node_ids(project, centroid_node_ids: Sequence[int]) -> None:
    """
    Mark the given nodes as centroids (``is_centroid=1``); all others set to 0.

    Call ``project.network.build_graphs()`` afterward to refresh ``project.network.graphs``.
    """
    validate_aequilibrae_project_network(project)

    ids = [int(x) for x in centroid_node_ids]
    id_set = set(ids)
    conn = getattr(project, "conn", None) or getattr(project, "db_connection", None)
    if conn is None:
        raise RuntimeError("Project has no database connection.")

    def _run(db):
        db.execute("UPDATE nodes SET is_centroid = 0")
        for nid in ids:
            db.execute("UPDATE nodes SET is_centroid = 1 WHERE node_id = ?", [nid])
        if hasattr(db, "commit"):
            db.commit()

    def _verify(db):
        missing = [
            nid
            for nid in id_set
            if db.execute("SELECT 1 FROM nodes WHERE node_id=?", [nid]).fetchone() is None
        ]
        if missing:
            raise RuntimeError(
                "These centroid node_ids are not in the project's ``nodes`` table "
                f"(regenerate the AequilibraE project from the same GMNS as node.csv): "
                f"{missing[:25]}{'...' if len(missing) > 25 else ''}"
            )

    if hasattr(conn, "__enter__"):
        with conn as db:
            _run(db)
            _verify(db)
    else:
        _run(conn)
        _verify(conn)

    _coerce_links_endpoints_sqlite(project)
    project.network.build_graphs()


def extract_taz_submatrix(
    demand: np.ndarray,
    zone_ids: Sequence[int],
    keep_zones: Sequence[int],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract a square OD submatrix for a subset of TAZs (same order as ``keep_zones``).

    Parameters
    ----------
    demand :
        Full matrix, shape ``(len(zone_ids), len(zone_ids))``. Row ``i`` and column ``j``
        correspond to ``zone_ids[i]`` and ``zone_ids[j]``.
    zone_ids :
        Labels for current row/column ordering (length must match matrix extent).
    keep_zones :
        TAZ IDs to retain (each must appear in ``zone_ids``).

    Returns
    -------
    sub :
        Shape ``(len(keep_zones), len(keep_zones))``.
    keep_zones_array :
        The requested zones as an array (same order as input).
    """
    zlist = list(zone_ids)
    index = {z: i for i, z in enumerate(zlist)}
    missing = [z for z in keep_zones if z not in index]
    if missing:
        raise KeyError(f"TAZs not found in zone_ids: {missing}")

    keep = np.asarray(list(keep_zones), dtype=np.int64)
    rows = [index[int(z)] for z in keep]
    cols = rows
    sub = demand[np.ix_(rows, cols)]
    return sub, keep


def total_od_matrix_taz_order_from_long_csv(
    path: Union[str, Path],
    taz_order: Sequence[int],
) -> np.ndarray:
    """
    Build a dense OD matrix with rows/cols in ``taz_order`` from a long CSV
    (``zone_id_from``, ``zone_id_to``, ``demand``). Rows with zones not in ``taz_order`` are skipped.
    """
    path = Path(path)
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    need = {"zone_id_from", "zone_id_to", "demand"}
    if not need.issubset(df.columns):
        raise ValueError(f"{path} needs columns {sorted(need)}; found {list(df.columns)}")
    Z = len(taz_order)
    idx = {int(z): i for i, z in enumerate(taz_order)}
    mat = np.zeros((Z, Z), dtype=np.float64)
    for _, row in df.iterrows():
        o = int(row["zone_id_from"])
        d = int(row["zone_id_to"])
        if o not in idx or d not in idx:
            continue
        mat[idx[o], idx[d]] = float(row["demand"])
    return mat


def reorder_skim_to_taz_order(
    skim_cc: np.ndarray,
    graph_centroids: np.ndarray,
    centroid_pick_in_taz_order: np.ndarray,
) -> np.ndarray:
    """
    AequilibraE skim matrices are indexed in ``graph_centroids`` (DB) order. OD and logit
    use **TAZ** order (:data:`TAZ_SUBSET_EIGHT`). This maps skim[i,j] from centroid-index
    space to TAZ-index space using the same ``node_id`` per TAZ as ``pick_centroid_node_ids_per_zone_from_node_csv``.
    """
    gc = np.asarray(graph_centroids).ravel()
    pick = np.asarray(centroid_pick_in_taz_order, dtype=np.int64).ravel()
    idx_map = {int(c): i for i, c in enumerate(gc)}
    Z = len(pick)
    out = np.full((Z, Z), np.inf, dtype=np.float64)
    for i in range(Z):
        for j in range(Z):
            oi = idx_map[int(pick[i])]
            oj = idx_map[int(pick[j])]
            out[i, j] = skim_cc[oi, oj]
    return out


def skim_travel_time_matrix(
    graph,
    *,
    time_field: str,
    block_centroid_flows: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    All-or-nothing skim of **one** cost field (``time_field``) between all centroids.

    Returns
    -------
    skim_ZZ : ndarray
        Shape ``(Z, Z)`` in **graph.centroids** order (same as ``SkimResults.matrix_view``).
    graph_centroids : ndarray
        Centroid node ids for that ordering.
    """
    from aequilibrae.paths import NetworkSkimming

    graph.set_blocked_centroid_flows(block_centroid_flows)
    graph.set_graph(time_field)
    graph.set_skimming([time_field])
    skm = NetworkSkimming(graph)
    skm.execute()
    mat = skm.results.skims.matrix_view
    if mat.ndim == 3:
        skim_2d = mat[:, :, 0]
    else:
        skim_2d = mat
    gc = graph.centroids
    return np.asarray(skim_2d, dtype=np.float64), np.asarray(gc)


def skim_graph_matrices(
    graph,
    skim_fields: Sequence[str],
    *,
    cost_field: str,
    block_centroid_flows: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Skim several link fields along **shortest paths** with respect to ``cost_field`` (e.g. time).

    Returns
    -------
    skim : ndarray
        Shape ``(Z, Z, K)`` with ``K`` equal to the number of skim fields after prepending
        ``cost_field`` if it was missing from ``skim_fields``, in **graph.centroids** order.
    graph_centroids : ndarray
    """
    from aequilibrae.paths import NetworkSkimming

    fields = list(skim_fields)
    if cost_field not in fields:
        fields = [cost_field] + fields

    graph.set_blocked_centroid_flows(block_centroid_flows)
    graph.set_graph(cost_field)
    graph.set_skimming(fields)
    skm = NetworkSkimming(graph)
    skm.execute()
    mat = skm.results.skims.matrix_view
    if mat.ndim == 2:
        mat = mat[:, :, np.newaxis]
    gc = graph.centroids
    return np.asarray(mat, dtype=np.float64), np.asarray(gc)


def logit_mode_shares(utilities: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """
    Multinomial logit choice probabilities from systematic utilities.

    Parameters
    ----------
    utilities :
        Array whose last axis is mode, e.g. shape ``(..., M)`` for ``M`` modes.
    scale :
        Positive multiplier on utilities before exp (larger → sharper choices).

    Returns
    -------
    shares
        Same shape as ``utilities``; last axis sums to 1 (NaN rows propagate).
    """
    if scale <= 0:
        raise ValueError("scale must be positive")
    U = np.asarray(utilities, dtype=np.float64) * scale
    max_u = np.nanmax(U, axis=-1, keepdims=True)
    exp_u = np.exp(np.clip(U - max_u, -700, 700))
    denom = np.sum(exp_u, axis=-1, keepdims=True)
    denom = np.where(denom == 0, np.nan, denom)
    return exp_u / denom


def allocate_total_demand_to_modes(
    total_od: np.ndarray,
    utilities_odm: np.ndarray,
    scale: float = 1.0,
) -> np.ndarray:
    """
    Split total person trips by OD into mode-specific OD matrices using logit shares.

    Parameters
    ----------
    total_od :
        Shape ``(Z, Z)`` (person trips or vehicle trips for the *total* before split).
    utilities_odm :
        Shape ``(Z, Z, M)`` — systematic utility for each mode (e.g. order car, bus, ION).
    scale :
        Logit scale on utilities.

    Returns
    -------
    mode_od
        Shape ``(Z, Z, M)``. Sums over the last axis equal ``total_od`` where shares
        are finite.
    """
    if total_od.shape != utilities_odm.shape[:2]:
        raise ValueError("total_od and utilities_odm[:2] must match")
    shares = logit_mode_shares(utilities_odm, scale=scale)
    return total_od[..., np.newaxis] * shares


# So ``project.network.build_graphs()`` is safe even when a_node/b_node dtypes are non-numeric.
_install_aequilibrae_build_graphs_patch()


__all__ = [
    "TAZ_SUBSET_EIGHT",
    "DEFAULT_BUS_NODE_CSV",
    "DEFAULT_ION_NODE_CSV",
    "DEFAULT_ROAD_NODE_CSV",
    "MultiLayerPaths",
    "open_layer_project",
    "open_all_layers",
    "get_graph_for_mode",
    "graph_centroids",
    "create_memory_od_matrix",
    "fill_od_from_long_table",
    "suggest_cost_and_capacity_fields",
    "build_traffic_assignment",
    "pick_centroid_node_ids_per_zone_from_node_csv",
    "extract_assignment_submatrix_eight_zone",
    "validate_aequilibrae_project_network",
    "set_centroids_by_node_ids",
    "extract_taz_submatrix",
    "logit_mode_shares",
    "allocate_total_demand_to_modes",
    "total_od_matrix_taz_order_from_long_csv",
    "reorder_skim_to_taz_order",
    "skim_travel_time_matrix",
    "skim_graph_matrices",
]
