"""
Create and manage an AequilibraE project for the ION multimodal network (bus + LRT).
Use this so the ION network can be merged with the road network for traffic assignment.
"""
import os
import shutil

from .build_ion_network import build_nodes_and_links, export_to_gmns
from .config import BUS_GTFS_DIR, LRT_GTFS_DIR, ION_STOPS_CSV, ION_ROUTES_CSV, PROJECT_CRS


def _fill_aequilibrae_ba_columns(project, direction_columns=None):
    """Fill NULL _ab/_ba columns with 0 so build_graphs does not fail."""
    if direction_columns is None:
        direction_columns = (
            "lanes_ab", "lanes_ba", "speed_ab", "speed_ba",
            "capacity_ab", "capacity_ba", "travel_time_ab", "travel_time_ba",
        )
    links_data = project.network.links.data
    to_fill = [c for c in direction_columns if c in links_data.columns]
    if not to_fill:
        return
    conn = getattr(project, "conn", None) or getattr(project, "db_connection", None)
    if conn is None:
        return
    try:
        if hasattr(conn, "__enter__"):
            with conn as c:
                for col in to_fill:
                    c.execute(f"UPDATE links SET {col} = 0 WHERE {col} IS NULL OR {col} = ''")
                if hasattr(c, "commit"):
                    c.commit()
        else:
            for col in to_fill:
                conn.execute(f"UPDATE links SET {col} = 0 WHERE {col} IS NULL OR {col} = ''")
            if hasattr(conn, "commit"):
                conn.commit()
    except Exception:
        pass


def create_aequilibrae_project(
    project_path=None,
    bus_gtfs=None,
    lrt_gtfs=None,
    ion_stops_csv=None,
    ion_routes_csv=None,
    nodes_df=None,
    links_df=None,
    srid=None,
    overwrite=True,
    verbose=True,
):
    """
    Build the ION (bus + LRT) network, export to GMNS, and create an AequilibraE project.

    Parameters
    ----------
    project_path : str, optional
        Directory for the AequilibraE project. Default: ion_network/aequilibrae_project.
    bus_gtfs : str, optional
        Path to bus GTFS directory. Default: Raw_GTFS from config.
    lrt_gtfs : str, optional
        Path to LRT GTFS directory. Default: GTFS(onlyLRT) from config.
    ion_stops_csv : str, optional
        Path to ION_Stops.csv.
    ion_routes_csv : str, optional
        Path to ION_Routes.csv.
    nodes_df : pandas.DataFrame, optional
        Pre-built nodes (skip build if provided with links_df).
    links_df : pandas.DataFrame, optional
        Pre-built links (skip build if provided with nodes_df).
    srid : int, optional
        SRID for network coordinates (default from PROJECT_CRS, typically 26917).
    overwrite : bool
        If True, remove existing project_path before creating.
    verbose : bool
        If True, print progress.

    Returns
    -------
    project : aequilibrae.Project
        Open project with network loaded and graphs built.
    nodes_df : pandas.DataFrame
        ION network nodes.
    links_df : pandas.DataFrame
        ION network links.
    """
    from aequilibrae import Project

    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_path = project_path or os.path.join(base_dir, "aequilibrae_project")
    bus_gtfs = bus_gtfs or BUS_GTFS_DIR
    lrt_gtfs = lrt_gtfs or LRT_GTFS_DIR
    ion_stops_csv = ion_stops_csv or ION_STOPS_CSV
    ion_routes_csv = ion_routes_csv or ION_ROUTES_CSV

    # Parse SRID from PROJECT_CRS (e.g., "EPSG:26917" -> 26917)
    if srid is None and PROJECT_CRS:
        srid = int(PROJECT_CRS.split(":")[-1]) if ":" in str(PROJECT_CRS) else 26917
    srid = srid or 26917

    if overwrite and os.path.exists(project_path):
        shutil.rmtree(project_path)

    if nodes_df is None or links_df is None:
        if verbose:
            print("\n=== Building ION Network (Bus + LRT) ===")
        nodes_df, links_df = build_nodes_and_links(
            bus_gtfs=bus_gtfs,
            lrt_gtfs=lrt_gtfs,
            ion_stops_csv=ion_stops_csv,
            ion_routes_csv=ion_routes_csv,
        )

    gmns_dir = os.path.join(base_dir, "data", "gmns")
    if verbose:
        print("\n=== Exporting to GMNS ===")
    node_file, link_file, geometry_file = export_to_gmns(nodes_df, links_df, gmns_dir, srid=srid)

    project = Project()
    project.new(project_path)
    if verbose:
        print("✓ Project created successfully")

    if verbose:
        print("Importing network from GMNS...")
    project.network.create_from_gmns(
        link_file_path=link_file,
        node_file_path=node_file,
        geometry_path=geometry_file,
        srid=srid,
    )

    if verbose:
        print(f"Number of links imported: {len(project.network.links.data)}")

    # Set travel_time on links for graph-based shortest path and assignment
    conn = getattr(project, "conn", None) or getattr(project, "db_connection", None)
    link_cols = set(project.network.links.data.columns)
    has_ab = "travel_time_ab" in link_cols
    has_tt = "travel_time" in link_cols

    if conn is not None:
        try:
            def _run(db):
                if has_tt:
                    try:
                        db.execute("SELECT travel_time FROM links LIMIT 1")
                    except Exception:
                        db.execute("ALTER TABLE links ADD COLUMN travel_time NUMERIC")
                params = [(float(row["travel_time_min"]), int(row["link_id"])) for _, row in links_df.iterrows()]
                if has_ab:
                    db.executemany("UPDATE links SET travel_time_ab = ? WHERE link_id = ?", params)
                if has_tt:
                    db.executemany("UPDATE links SET travel_time = ? WHERE link_id = ?", params)
                if hasattr(db, "commit"):
                    db.commit()

            if hasattr(conn, "__enter__"):
                with conn as db:
                    _run(db)
            else:
                _run(conn)
            if verbose:
                print("  Set travel_time_ab and travel_time on links from built network.")
        except Exception as e:
            if verbose:
                print(f"  Warning: could not set travel times: {e}")

    _fill_aequilibrae_ba_columns(project)
    project.network.build_graphs()
    if verbose:
        print("✓ Network built successfully!")
        print(f"  Nodes: {project.network.count_nodes()}")
        print(f"  Links: {project.network.count_links()}")

    return project, nodes_df, links_df
