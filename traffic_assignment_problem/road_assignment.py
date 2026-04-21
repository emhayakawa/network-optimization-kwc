"""
Helpers for static traffic assignment on the **road** AequilibraE project.

Centroids are nodes with ``is_centroid = 1``; OD matrices must align with
``graph.centroids`` (see :func:`create_memory_od_matrix`).

Example (after centroids and demand exist)::

    from road_assignment import (
        RoadNetworkPaths,
        open_layer_project,
        get_graph_for_mode,
        create_memory_od_matrix,
        build_traffic_assignment,
    )

    paths = RoadNetworkPaths()
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

DEFAULT_ROAD_NODE_CSV: Path = _REPO_ROOT / "road_network" / "data" / "gmns" / "node.csv"


@dataclass
class RoadNetworkPaths:
    """Default location of the road AequilibraE project folder in this repo."""

    road_project: Path = _REPO_ROOT / "road_network" / "aequilibrae_project"


def open_layer_project(project_dir: Union[str, Path]):
    """
    Open an existing AequilibraE project from disk.

    Build the project first with ``road_network.build_road_network`` (``create_aequilibrae=True``).
    """
    from aequilibrae import Project

    project_dir = Path(project_dir)
    if not project_dir.is_dir():
        raise FileNotFoundError(f"AequilibraE project not found: {project_dir}")
    project = Project()
    project.open(str(project_dir))
    return project


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
    Pick one ``node_id`` per TAZ from a GMNS-style ``node.csv``.

    Use this to decide **which** network nodes become AequilibraE **centroids** for your
    eight-zone (or other) subset.

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
            "``road_network.build_network(create_aequilibrae=True)``)."
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


# So ``project.network.build_graphs()`` is safe even when a_node/b_node dtypes are non-numeric.
_install_aequilibrae_build_graphs_patch()

__all__ = [
    "TAZ_SUBSET_EIGHT",
    "DEFAULT_ROAD_NODE_CSV",
    "RoadNetworkPaths",
    "open_layer_project",
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
]
