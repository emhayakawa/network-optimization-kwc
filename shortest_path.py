"""
Helper function for testing shortest path computation in AequilibraE
"""


def test_shortest_path(project, orig_node, dest_node, verbose=True):
    """
    Test shortest path computation between two nodes in an AequilibraE project.

    Parameters:
    -----------
    project : aequilibrae.Project
        The AequilibraE project with built network graphs
    orig_node : int
        Origin node ID
    dest_node : int
        Destination node ID
    verbose : bool, optional
        Whether to print detailed output (default: True)

    Returns:
    --------
    dict
        Dictionary containing path results:
        - 'path_nodes': list of node IDs in the path
        - 'path_links': list of link IDs in the path
        - 'total_distance': total distance in meters
        - 'found': boolean indicating if path was found
    """
    if verbose:
        print(f"Finding shortest path from node {orig_node} to node {dest_node}...")

    # Use the graph already built by project.network.build_graphs()
    graph_modes = list(project.network.graphs.keys())
    if not graph_modes:
        raise RuntimeError("No graphs built. Check network and build_graphs().")
    graph = project.network.graphs.get("c") or project.network.graphs[graph_modes[0]]
    if graph != project.network.graphs.get("c"):
        if verbose:
            print(f"  Using graph mode: {graph_modes[0]} (no 'c' mode)")
    graph.set_graph("length")
    graph.set_blocked_centroid_flows(False)

    if verbose:
        print("✓ Graph configured (cost = length)")

    # Compute shortest path; returns a PathResults-like object
    res = graph.compute_path(orig_node, dest_node)

    # Check results (res.path = link IDs, res.path_nodes = node sequence, res.milepost = cumulative distance)
    if res.path is not None and len(res.path) > 0:
        path_nodes = res.path_nodes
        path_links = res.path
        total_distance = res.milepost[-1] if len(res.milepost) > 0 else 0.0

        if verbose:
            print(f"✓ Shortest path found!")
            print(f"  Total distance: {total_distance:.2f} meters")
            print(f"  Number of nodes: {len(path_nodes)}")
            print(f"  Number of links: {len(path_links)}")
            print(f"  Path (node IDs): {path_nodes[:20]}...")

        return {
            "path_nodes": path_nodes,
            "path_links": path_links,
            "total_distance": total_distance,
            "found": True,
        }
    else:
        if verbose:
            print("✗ No path found between these nodes!")

        return {
            "path_nodes": None,
            "path_links": None,
            "total_distance": None,
            "found": False,
        }
