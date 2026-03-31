#!/usr/bin/env python3
"""
Build the ION (bus + LRT) network and create ``ion_network/aequilibrae_project``.

Run from the URA repository root::

    python ion_network/create_aequilibrae_project.py

Requires ``Data/Raw_GTFS``, ``Data/GTFS(withLRT)``, ``Data/ION_Stops.csv``,
``Data/ION_Routes.csv`` (paths from ``ion_network.config``).

Options::

    python ion_network/create_aequilibrae_project.py --no-overwrite   # fail if project exists
"""
from __future__ import annotations

import argparse
import os
import sys

_URA_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _URA_ROOT not in sys.path:
    sys.path.insert(0, _URA_ROOT)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create AequilibraE project for ION network.")
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Do not delete an existing aequilibrae_project folder first.",
    )
    parser.add_argument("--quiet", action="store_true", help="Less console output.")
    args = parser.parse_args()

    from ion_network.aequilibrae_network import create_aequilibrae_project

    project, nodes_df, links_df = create_aequilibrae_project(
        overwrite=not args.no_overwrite,
        verbose=not args.quiet,
    )
    try:
        project.close()
    except Exception:
        pass

    if not args.quiet:
        out = os.path.join(os.path.dirname(__file__), "aequilibrae_project")
        print(f"\nDone. Project directory: {out}")
        print(f"  Nodes: {len(nodes_df)}, Links: {len(links_df)}")


if __name__ == "__main__":
    main()
