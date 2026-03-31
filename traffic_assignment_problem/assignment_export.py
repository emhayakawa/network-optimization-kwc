"""
Export TAP assignment results to GeoPackage for ArcGIS / QGIS.

* **assignment_links** — **all** network link geometries **left-joined** to ``assig.results()`` (``demand_tot``,
  PCE_tot, congested times, etc.). Unused links still appear with zero / null assignment fields—filter
  in GIS (e.g. ``demand_tot > 0``) or use ``export_assignment_gpkg(..., assignment_links_flow_only=True)``.
  Sum of ``demand_tot`` over links is not total OD trips (trips cross many links).

* **path_links** / **path_nodes** (optional) — shortest paths on the *post-assignment* cost field
  ``Congested_Time_Max`` when available, else the graph time field used for TAP. One row per OD pair
  requested via ``--export-path O:D``.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, List, Literal, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from aequilibrae.paths import Graph

_REPO_ROOT = Path(__file__).resolve().parent.parent

LayerName = Literal["road", "ion"]


def _ion_data_paths() -> Tuple[Path, Path, Path]:
    base = _REPO_ROOT / "ion_network" / "data"
    return base / "node.csv", base / "link.csv", base / "geometry.csv"


def _road_gmns_paths() -> Tuple[Path, Path, Path]:
    base = _REPO_ROOT / "road_network" / "data" / "gmns"
    return base / "node.csv", base / "link.csv", base / "geometry.csv"


def _links_gdf_from_csv(
    node_csv: Path,
    link_csv: Path,
    geometry_csv: Path,
    *,
    crs_proj: str = "EPSG:26917",
) -> "geopandas.GeoDataFrame":
    """Build a links GeoDataFrame from saved GMNS-style CSVs (same logic as ion export_to_arcgis_from_data)."""
    from shapely import wkt
    import geopandas as gpd

    nodes_df = pd.read_csv(node_csv)
    links_df = pd.read_csv(link_csv)
    geom_df = pd.read_csv(geometry_csv)
    geom_lookup = geom_df.set_index("geometry_id")["geometry"].to_dict()
    link_geoms = []
    for _, row in links_df.iterrows():
        wkt_str = geom_lookup.get(row.get("geometry_id", row.get("link_id")))
        link_geoms.append(wkt.loads(wkt_str) if wkt_str else None)
    links_df = links_df.copy()
    links_df["geometry"] = link_geoms
    gdf = gpd.GeoDataFrame(
        links_df.dropna(subset=["geometry"]),
        geometry="geometry",
        crs=crs_proj,
    )
    return gdf.to_crs("EPSG:4326")


def load_links_geodataframe(layer: LayerName):
    """
    Load link geometries for ``layer``: prefer ``arcgis_export/*_network.gpkg`` layer ``links``,
    else build from saved ``data`` CSVs.
    """
    import geopandas as gpd

    gpkg = _REPO_ROOT / f"{layer}_network" / "arcgis_export" / f"{layer}_network.gpkg"
    if gpkg.is_file():
        return gpd.read_file(gpkg, layer="links")

    if layer == "ion":
        n, l, g = _ion_data_paths()
    else:
        n, l, g = _road_gmns_paths()
    for p in (n, l, g):
        if not p.is_file():
            raise FileNotFoundError(
                f"Missing {p}. Build the {layer} network (ArcGIS export or GMNS data folder) first."
            )
    return _links_gdf_from_csv(n, l, g)


def load_nodes_geodataframe(layer: LayerName):
    """Point layer for path node export (optional)."""
    import geopandas as gpd

    gpkg = _REPO_ROOT / f"{layer}_network" / "arcgis_export" / f"{layer}_network.gpkg"
    if gpkg.is_file():
        return gpd.read_file(gpkg, layer="nodes")

    if layer == "ion":
        node_csv = _ion_data_paths()[0]
    else:
        node_csv = _road_gmns_paths()[0]
    if not node_csv.is_file():
        raise FileNotFoundError(f"Missing {node_csv}")
    nodes_df = pd.read_csv(node_csv)
    return gpd.GeoDataFrame(
        nodes_df,
        geometry=gpd.points_from_xy(nodes_df["x_coord"], nodes_df["y_coord"]),
        crs="EPSG:26917",
    ).to_crs("EPSG:4326")


def merge_assignment_results_to_links(
    links_gdf,
    results_df: pd.DataFrame,
) -> "geopandas.GeoDataFrame":
    """Left-join assignment columns onto link geometries (``link_id``)."""
    import geopandas as gpd

    res = results_df.copy()
    if res.index.name != "link_id" and "link_id" not in res.columns:
        res = res.reset_index()
    else:
        res = res.reset_index()
    if "link_id" not in res.columns:
        raise ValueError("Assignment results must have link_id as index or column.")
    res["link_id"] = res["link_id"].astype(int)
    lg = links_gdf.copy()
    lg["link_id"] = lg["link_id"].astype(int)
    merged = lg.merge(res, on="link_id", how="left", suffixes=("", "_assig"))
    return gpd.GeoDataFrame(merged, geometry="geometry", crs=links_gdf.crs)


def _apply_path_cost_column(graph: "Graph", results_df: pd.DataFrame, time_field: str) -> str:
    """Add ``_tap_path_cost`` from Congested_Time_Max or fall back to graph time field."""
    gdf = graph.graph.copy()
    lid = gdf["link_id"].to_numpy()
    if "Congested_Time_Max" in results_df.columns:
        ct = results_df["Congested_Time_Max"].reindex(lid).to_numpy()
        base = gdf[time_field].to_numpy(dtype=np.float64)
        use = np.where(np.isfinite(ct) & (ct > 0), ct, base)
    else:
        use = gdf[time_field].to_numpy(dtype=np.float64)
    gdf["_tap_path_cost"] = use
    graph.graph = gdf
    graph.set_graph("_tap_path_cost")
    return "_tap_path_cost"


def path_nodes_to_link_rows(
    path_nodes: np.ndarray,
    links_gdf,
):
    """Match consecutive node pairs to link rows (first match on directed from→to)."""
    import geopandas as gpd

    if path_nodes is None or len(path_nodes) < 2:
        return links_gdf.iloc[0:0].copy()
    pairs = links_gdf.set_index(["from_node_id", "to_node_id"], drop=False)
    rows = []
    for i in range(len(path_nodes) - 1):
        f, t = int(path_nodes[i]), int(path_nodes[i + 1])
        try:
            r = pairs.loc[(f, t)]
            row = r.iloc[0] if isinstance(r, pd.DataFrame) else r
            rows.append(row)
        except KeyError:
            continue
    if not rows:
        return links_gdf.iloc[0:0].copy()
    return gpd.GeoDataFrame(pd.DataFrame(rows), geometry="geometry", crs=links_gdf.crs)


def export_paths_to_gpkg(
    links_gdf,
    graph: "Graph",
    time_field: str,
    results_df: pd.DataFrame,
    od_pairs: Sequence[Tuple[int, int]],
    out_path: Path,
    *,
    nodes_gdf=None,
) -> None:
    """
    For each (origin_node, dest_node), compute one shortest path using post-assignment costs
    when ``Congested_Time_Max`` is present, and append ``path_links`` (+ ``path_nodes`` if nodes_gdf given)
    to an **existing** GeoPackage (does not remove other layers).
    """
    import geopandas as gpd
    from aequilibrae.paths.results import PathResults

    out_path = Path(out_path)
    if not out_path.is_file():
        raise FileNotFoundError(f"Expected existing GeoPackage to append paths: {out_path}")

    _apply_path_cost_column(graph, results_df, time_field)
    pr = PathResults()
    pr.prepare(graph)

    path_link_parts: List[pd.DataFrame] = []
    used_nodes: set[int] = set()
    for o, d in od_pairs:
        pr.compute_path(int(o), int(d), early_exit=True)
        pn = pr.path_nodes
        if pn is None or len(pn) < 2:
            continue
        for nid in pn:
            used_nodes.add(int(nid))
        seg = path_nodes_to_link_rows(pn, links_gdf)
        if seg is None or len(seg) == 0:
            continue
        seg = seg.copy()
        seg["path_o"] = int(o)
        seg["path_d"] = int(d)
        seg["od_pair"] = f"{o}:{d}"
        path_link_parts.append(seg)

    if path_link_parts:
        all_pl = gpd.GeoDataFrame(
            pd.concat(path_link_parts, ignore_index=True),
            geometry="geometry",
            crs=links_gdf.crs,
        )
        all_pl.to_file(out_path, layer="path_links", driver="GPKG", mode="a")
    else:
        gpd.GeoDataFrame(geometry=[], crs=links_gdf.crs).to_file(out_path, layer="path_links", driver="GPKG", mode="a")

    try:
        graph.graph.drop(columns=["_tap_path_cost"], errors="ignore", inplace=True)
        graph.set_graph(time_field)
    except Exception:
        pass

    if nodes_gdf is not None and used_nodes:
        sub = nodes_gdf[nodes_gdf["node_id"].isin(list(used_nodes))].copy()
        if len(sub):
            sub.to_file(out_path, layer="path_nodes", driver="GPKG", mode="a")


def export_assignment_gpkg(
    layer: LayerName,
    results_df: pd.DataFrame,
    out_path: Path,
    *,
    graph: Optional["Graph"] = None,
    time_field: Optional[str] = None,
    path_od_pairs: Optional[Iterable[Tuple[int, int]]] = None,
    assignment_links_flow_only: bool = False,
) -> Path:
    """
    Write ``assignment_links`` to ``out_path``; optionally append ``path_links`` / ``path_nodes``.

    Parameters
    ----------
    path_od_pairs :
        Centroid (or any) node pairs ``(origin, dest)`` for sample shortest-path overlays.
    assignment_links_flow_only :
        If True, drop links where ``demand_tot`` is missing or ≤ 0 (full network kept when False).
    """
    out_path = Path(out_path)
    os.makedirs(out_path.parent, exist_ok=True)
    links_gdf = load_links_geodataframe(layer)
    merged = merge_assignment_results_to_links(links_gdf, results_df)
    if assignment_links_flow_only:
        if "demand_tot" not in merged.columns:
            import warnings

            warnings.warn(
                "assignment_links_flow_only requested but merged layer has no demand_tot; writing all links."
            )
        else:
            n0 = len(merged)
            merged = merged[merged["demand_tot"].fillna(0) > 1e-9].copy()
            n1 = len(merged)
            if n1 < n0:
                print(
                    f"  assignment_links: {n1} links with demand_tot > 0 (of {n0} in base network)"
                )
    if out_path.exists():
        out_path.unlink()
    merged.to_file(out_path, layer="assignment_links", driver="GPKG")

    pairs = list(path_od_pairs) if path_od_pairs else []
    if pairs and graph is not None and time_field:
        try:
            nodes_gdf = load_nodes_geodataframe(layer)
        except Exception:
            nodes_gdf = None
        try:
            export_paths_to_gpkg(
                links_gdf,
                graph,
                time_field,
                results_df,
                pairs,
                out_path,
                nodes_gdf=nodes_gdf,
            )
        except Exception as e:
            import warnings

            warnings.warn(f"path_links/path_nodes export failed (assignment_links still saved): {e}")
    return out_path
