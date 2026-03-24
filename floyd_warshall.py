import os
import numpy as np
import pandas as pd
from aequilibrae import Project

# --- Load your AequilibraE project (run from URA directory) ---
project_path = "Network/aequilibrae_project"
if not os.path.exists(project_path):
    raise FileNotFoundError(
        f"AequilibraE project not found at '{project_path}'. "
        "Run test.py first to create the project, then run this script from the URA directory."
    )
project = Project()
project.open(project_path)

# --- Extract links from the project ---
links_df = project.network.links.data
# AequilibraE may use a_node/b_node (internal) or from_node_id/to_node_id (e.g. after GMNS import)
node_a = "a_node" if "a_node" in links_df.columns else "from_node_id"
node_b = "b_node" if "b_node" in links_df.columns else "to_node_id"
weight_col = "length" if "length" in links_df.columns else "distance"
for col in [node_a, node_b, weight_col]:
    if col not in links_df.columns:
        raise KeyError(
            f"Links table missing column '{col}'. Available: {links_df.columns.tolist()}"
        )

edges = links_df[[node_a, node_b, weight_col]].copy()
edges = edges.rename(columns={node_a: "a_node", node_b: "b_node"})
edges = edges.dropna(subset=["a_node", "b_node", weight_col])

if len(edges) == 0:
    raise ValueError(
        f"No links found after dropna. Links table has {len(links_df)} rows; "
        f"columns: {links_df.columns.tolist()}. "
        "Check that the project was built by test.py and contains a network."
    )

# --- Build node list and distance matrix ---
nodes = sorted(set(edges["a_node"].astype(int)) | set(edges["b_node"].astype(int)))
node_to_idx = {n: i for i, n in enumerate(nodes)}
n_nodes = len(nodes)

dist = np.full((n_nodes, n_nodes), np.inf)
np.fill_diagonal(dist, 0.0)
for _, row in edges.iterrows():
    i, j = node_to_idx[int(row["a_node"])], node_to_idx[int(row["b_node"])]
    w = float(row[weight_col])
    dist[i, j] = min(dist[i, j], w)

# --- Floyd-Warshall ---
for k in range(n_nodes):
    for i in range(n_nodes):
        for j in range(n_nodes):
            if dist[i, k] + dist[k, j] < dist[i, j]:
                dist[i, j] = dist[i, k] + dist[k, j]

# --- Check connectivity ---
idx_to_node = {i: n for n, i in node_to_idx.items()}
unconnected_pairs = [
    (idx_to_node[i], idx_to_node[j])
    for i in range(n_nodes)
    for j in range(n_nodes)
    if i != j and not np.isfinite(dist[i, j])
]
if n_nodes == 0:
    raise ValueError(
        "No nodes in graph. Links table may use different column names or have no valid links. "
        f"Link columns seen: {links_df.columns.tolist()}; rows before dropna: {len(links_df)}."
    )
print(f"Nodes: {n_nodes}")
if not unconnected_pairs:
    print("✓ All node pairs are connected (strongly connected).")
else:
    print(f"✗ {len(unconnected_pairs)} node pairs have no path (sample: {unconnected_pairs[:5]}).")

# --- Close project ---
project.close()
