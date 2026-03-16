"""
Create and manage an AequilibraE project for the bus network.
Use this so shortest path and (later) traffic assignment use the same engine.
"""
import os
import shutil

from .build_bus_network import build_nodes_and_links, export_to_gmns
from .config import DEFAULT_GTFS_PATH


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
    srid=26917,
    overwrite=True,
    verbose=True,
):
    """
    Build the bus network from GTFS, export to GMNS, and create an AequilibraE project
    with travel_time_ab set from GTFS (for time-based shortest path and assignment).

    Parameters
    ----------
    project_path : str, optional
        Directory for the AequilibraE project. Default: bus_network/aequilibrae_project.
    gtfs_path : str, optional
        Path to GTFS directory. Default: Data/Raw_GTFS from config.
    transfer_minutes : float, optional
        Transfer time in minutes (default from config).
    srid : int
        SRID for network coordinates (default 26917).
    overwrite : bool
        If True, remove existing project_path before creating.
    verbose : bool
        If True, print progress (same style as test.py).

    Returns
    -------
    project : aequilibrae.Project
        Open project with network loaded and graphs built.
    nodes_df : pandas.DataFrame
        Bus network nodes (for lookups).
    links_df : pandas.DataFrame
        Bus network links (for lookups).
    """
    from aequilibrae import Project

    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_path = project_path or os.path.join(base_dir, "aequilibrae_project")

    if overwrite and os.path.exists(project_path):
        shutil.rmtree(project_path)

    nodes_df, links_df = build_nodes_and_links(DEFAULT_GTFS_PATH)
    gmns_dir = os.path.join(base_dir, "data", "gmns")
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

    links_table = project.network.links
    if verbose:
        print(f"Number of links imported: {len(links_table.data)}")
        print(f"Columns in AequilibraE links table: {links_table.data.columns.tolist()}")
    if verbose and "length" in links_table.data.columns:
        print(f"✓ 'length' column found (min: {links_table.data['length'].min():.2f}, max: {links_table.data['length'].max():.2f})")
    elif verbose:
        print("Available numeric columns:", links_table.data.select_dtypes(include="number").columns.tolist())

    # Set travel time from our links (minutes) so graph can use it for time-based path/assignment.
    # project.conn is a context manager: we must use "with conn as db" and run SQL inside the block.
    conn = getattr(project, "conn", None) or getattr(project, "db_connection", None)
    link_cols = set(project.network.links.data.columns)
    has_ab = "travel_time_ab" in link_cols
    has_tt = "travel_time" in link_cols
    # Ensure "travel_time" column exists (graph uses this name); GMNS may not create it
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
