# URA — Road, bus, and ION networks (KWC)

This repository builds multimodal transportation networks for the Kitchener–Waterloo–Cambridge area, runs **shortest-path** routing with **generalized cost**, supports **zone-to-zone** routing (best path over all node pairs between two TAZs), and runs **static traffic assignment (TAP)** on the road network with AequilibraE.

Run Python scripts from the **repository root** unless noted otherwise.

### How this README is organized

For each **network package** (`road_network`, `bus_network`, `ion_network`), the story is told as a **single end-to-end pipeline**: numbered steps run from **raw or cleaned inputs** → **graph (nodes & links)** → **costs and routing** → **saved outputs**. Design choices (why something is filtered, clustered, or split across routes) are folded into the step where it happens, instead of a separate “design” block. **How to run** and **key files** come after the pipeline. Later sections (**transit** module, **zone-to-zone**, **TAP**) follow the same idea: **what happens, in order**, then commands.

---

## 1. `Data/` — inputs and what they are for

| Location | Role |
|----------|------|
| **`Data/Roads/`** (`Roads.shp` and sidecars) | Road centerlines and attributes (e.g. `CartoClass`, speed). Consumed by `road_network` to build the drivable graph, capacities, and GMNS for AequilibraE. |
| **`Data/Traffic_Lights_.../`** | Point layer of signalized intersections. Snapped to road nodes; used to add signal delay into travel time and thus **generalized cost** on links. |
| **`Data/2011 RMOW RTM TAZ_zone/`** | TAZ polygon shapefile. Used to assign **`zone_id`** to road and transit nodes (spatial join) so you can route **zone-to-zone** and align demand with TAZs. |
| **`Data/Raw_GTFS/`** | Standard GTFS (routes, stops, stop_times, trips, etc.). **Bus-only** network (`bus_network`) and **bus half** of the ION multimodal build (`ion_network`) read schedules and geometry from here. |
| **`Data/GTFS(withLRT)/`** | GTFS feed that includes **ION LRT** (light rail). Used by `ion_network` for LRT stop times and shapes alongside bus data. |
| **`Data/ION_Stops.csv`** | ION extension stops (e.g. Cambridge extension) in projected coordinates. Merged with LRT GTFS in `ion_network/preprocess_lrt.py`; drives **extension links** beyond current operating LRT in GTFS alone. |
| **`Data/ION_Routes.csv`** | Route/line geometry for ION (e.g. alignment). Used when building the multimodal network so links follow the correct paths. |

If any of these paths move, update the corresponding `config.py` under `road_network/`, `bus_network/`, or `ion_network/`.

---

## 2. `road_network/` — road graph build and outputs

**Purpose:** Turn KWC **road centerlines** into a **directed network** of nodes and links suitable for shortest-path routing (NetworkX + generalized cost) and for **TAP** in AequilibraE (car mode `c`).

### Pipeline (end-to-end)

1. **Load and filter roads** — Read `Data/Roads/`, reproject if needed, and keep only classes in **`ALLOWED_ROAD_CLASSES`** (`CartoClass`: e.g. freeway, highway, ramp, arterial). This focuses the model on higher-type roads; widen the list in `config.py` if you need locals/collectors.

2. **Draft intersection nodes** — Treat each road as a line between endpoints. Any coordinate shared by **two or more** segments becomes an **intersection node** (see `create_nodes_from_roads` in `nodes.py`). You are extracting a graph from geometry, not importing a ready-made junction layer.

3. **Traffic lights** — Load the traffic-signal points; each light **snaps** to the nearest node within **`SNAP_TOLERANCE_M`** or becomes a **new** node. Nodes pick up **`signalized = 1`** where a signal applies (`add_traffic_lights_to_nodes`). **`SIGNAL_DELAY_SECONDS`** in `config.py` (default 40 s) is the nominal delay constant for planning; link **`travel_time_min`** is still **length ÷ speed** unless you extend the model to add per-signal time into GC.

4. **Cluster nearby nodes (divided roads)** — Parallel centerlines for two directions often create **multiple** intersection points at one physical junction. Nodes within **`NODE_CLUSTER_TOLERANCE_M`** (default 30 m) are merged to **one** centroid node so the graph does not double-count the same intersection (`cluster_nearby_nodes` / `merge_clustered_nodes`).

5. **Build directed links** — For each road line, create **directed** edges between clustered endpoints using **`FlowDirect`** (two-way vs one-way). Each link gets length from geometry, speed from **`SpeedZone`**, **capacity** from class + lanes (for BPR/TAP), and travel time from length/speed (`links.py`).

6. **Keep the main component** — Drop small disconnected fragments; retain the **largest** weakly connected component so routing does not hit stray islands (`filter_to_main_component`).

7. **Optional TAZ labels** — If enabled, assign **`zone_id`** from the TAZ shapefile to each node for zone-to-zone workflows (`assign_zone_id` path in `build_road_network.py`).

8. **Generalized cost on links** — Add **`generalized_cost`**:  
   **GC = (distance_km × `COST_PER_KM`) + (travel_time_h × `VALUE_OF_TIME`)**  
   (`road_network/shortest_path.py`). Sample routing in the build script uses **NetworkX** Dijkstra on this column.

9. **Write GMNS and optional projects** — Save `data/gmns/` (`node.csv`, `link.csv`, `geometry.csv`). If **`create_aequilibrae=True`**, build **`road_network/aequilibrae_project/`** and graphs. If **`export_arcgis=True`**, write **`road_network/arcgis_export/road_network.gpkg`**.

10. **Optional test path** — `build_road_network.py` under `if __name__ == "__main__"` can run a sample shortest path and export to GeoPackage.

**Scale (this repo’s last build, order of magnitude):** ~**2,995** nodes and ~**6,130** directed links in `road_network/data/gmns/`; counts change when you rebuild or change filters.

### How to run

```bash
python road_network/build_road_network.py
```

**Shortest-path test OD** — In the same file, bottom of `if __name__ == "__main__":`, set **`orig_node`** and **`dest_node`** to valid **`node_id`** values (or use the commented `all_node_ids[...]` lines). The test calls `compute_shortest_path_networkx(..., weight_field='generalized_cost')`.

```python
if __name__ == "__main__":
    project, nodes_gdf, edges_gdf = build_network()

    print("\n=== Testing shortest path (NetworkX with generalized cost) ===")
    all_node_ids = list(nodes_gdf['node_id'].values)
    orig_node = int(1761)  # or: int(all_node_ids[0])
    dest_node = int(2657)  # or: int(all_node_ids[min(100, len(all_node_ids) - 1)])
```

### Key files

| File / folder | Purpose |
|---------------|---------|
| `config.py` | Paths, CRS, `ALLOWED_ROAD_CLASSES`, capacity (vphpl), GC and signal parameters. |
| `nodes.py`, `links.py` | Intersections, clustering, traffic lights, directed links. |
| `shortest_path.py` | `generalized_cost`, NetworkX / optional AequilibraE helpers, path export. |
| `build_road_network.py` | Orchestrates the pipeline above. |
| `data/gmns/` | GMNS CSVs for AequilibraE. |
| `aequilibrae_project/` | SQLite project when built. |
| `arcgis_export/` | `road_network.gpkg` and optional shortest-path outputs. |

---

## 3. `bus_network/` — GTFS bus graph only

**Purpose:** Turn **`Data/Raw_GTFS`** into a **transit line graph**: passengers move along **route-specific** nodes and can **transfer** between routes at stops.

### Pipeline (end-to-end)

1. **Load GTFS** — Read routes, stops, stop_times, trips (schedule is the source of truth; this is not a road-centerline network).

2. **Stop–route nodes and in-sequence links** — For each **physical stop**, create **one node per route** that serves it (**(stop_id, route_id)**). In-sequence links follow **trip order** with travel time from schedules and distance from coordinates (haversine). Staying on one line uses links on that route’s nodes; **changing routes** at a stop is modeled by jumping to another route’s node—so one platform can map to **several nodes**, which makes **inter-route transfers** explicit.

3. **Clustering and transfers** — Stops within **`STOP_CLUSTER_RADIUS_M`** are grouped for transfer and connectivity. **Transfer links** connect route-nodes at the same stop/cluster with a time penalty (**`TRANSFER_TIME_MINUTES`**, default 15 min)—a simple stand-in for walk + wait between lines.

4. **Zones** — Join node coordinates to **TAZ polygons** to set **`zone_id`** (`zones.py`) for zone-to-zone routing and mapping.

5. **Generalized cost (routing)** — Use **`transit/shortest_path.py`**:  
   **GC ≈ fare + (waiting_time + in-vehicle time) × value-of-time**  
   with parameters in `bus_network/config.py` (`FARE_CONSTANT`, `WAITING_TIME_MINUTES`, `VALUE_OF_TIME`). You can minimize **time** or **distance** instead.

6. **Save and export** — Write `bus_network/data/` (`node.csv`, `link.csv`, `geometry.csv`). Optionally **`bus_network/arcgis_export/bus_network.gpkg`**.

7. **Optional test in `build_bus_network.py`** — `__main__` runs a sample `shortest_path_transit` and can export a path.

**Scale (this repo):** on the order of ~**2,868** nodes and ~**7,309** links in `bus_network/data/`.

### How to run

```bash
python bus_network/build_bus_network.py
```

**Sample shortest-path OD (origin / destination nodes)** — After the build, the script runs a test path. Set **`orig_node`** and **`dest_node`** at the bottom of **`bus_network/build_bus_network.py`** in the **`if __name__ == "__main__":`** block (look for the `=== Testing shortest path` section; defaults are on the order of lines **833–834**). Values must be valid **`node_id`** entries from the built `nodes_df` / `data/node.csv`. Optional AequilibraE helpers: `bus_network/aequilibrae_network.py`.

### Key files

| File / folder | Purpose |
|---------------|---------|
| `config.py` | GTFS path, TAZ, transfer time, clustering, fare/VOT/wait. |
| `build_bus_network.py` | Full pipeline. |
| `data/` | Saved network CSVs. |
| `zones.py` | TAZ → `zone_id`. |
| `shortest_path_bus.py` | Re-exports `transit.shortest_path`. |
| `arcgis_export/` | `bus_network.gpkg`. |

---

## 4. `ion_network/` — bus + ION LRT + extension (multimodal)

**Purpose:** Same **stop–route + transfer** idea as `bus_network`, but **add** ION LRT from a second GTFS feed, **extension** stops/geometry from CSVs, and **bus ↔ LRT** transfers.

### Pipeline (end-to-end)

1. **Two schedule sources** — **`Data/Raw_GTFS`** for GRT bus; **`Data/GTFS(withLRT)`** for ION LRT patterns and times. Two feeds reflect how the data are maintained (bus-only vs includes LRT).

2. **Extension beyond GTFS** — **`ION_Stops.csv`** / **`ION_Routes.csv`** supply alignments and stops (e.g. extension phases) merged in **`preprocess_lrt.py`** so the rail layer matches planned geometry, not only a static GTFS cut.

   **ION extension travel times (no schedule for new segments yet):** The LRT feed’s **`stop_times`** are used to infer a typical **in-vehicle time between consecutive ION stops** on operating trips: for each trip, segment time = **next arrival minus current departure** (minutes), and only positive values are kept. **`calculate_avg_travel_time`** in **`preprocess_lrt.py`** takes the **mean** of all such segment samples across trips. That **average minutes-per-segment** is then applied as **`travel_time_min`** on every **extension** link—both the connectors from the last GTFS stop (e.g. Fairway) to the first extension stop and the **sequential links between extension stops**—so extension runtimes are consistent with observed ION headways rather than guessed per link. If no samples exist, the code falls back to a small default (e.g. ~2.5 min).

3. **Build multimodal nodes and links** — `build_ion_network.py` constructs **(stop, route, mode)**-style topology, in-sequence links, bus–bus transfers, and **multimodal** links where bus and LRT stops fall within **`MULTIMODAL_TRANSFER_RADIUS_M`** with **`TRANSFER_TIME_BUS_LRT`**.

4. **Mode-specific waiting** — Origin wait can differ for bus vs LRT (**`WAITING_TIME_BUS`** vs **`WAITING_TIME_LRT`**) in `ion_network/config.py`.

5. **Generalized cost** — Same **`transit/shortest_path.py`** machinery as bus with ION parameters (fare + time × VOT).

6. **Outputs** — `ion_network/data/` (including optional **`extension_links.csv`**, **`stops_augmented.csv`** depending on run). Optional **`ion_network/arcgis_export/ion_network.gpkg`**. Optional **`ion_network/aequilibrae_project/`** via **`create_aequilibrae_project.py`** / `aequilibrae_network.py`.

7. **Optional test** — `build_ion_network.py` `__main__` can run a sample path and export.

**Scale (this repo):** on the order of ~**2,907** nodes and ~**7,902** links in `ion_network/data/`.

### How to run

```bash
python ion_network/build_ion_network.py
python ion_network/create_aequilibrae_project.py
```

**Sample shortest-path OD (origin / destination nodes)** — After the build, **`build_ion_network.py`** can run a test `shortest_path_transit`. Set **`orig_node`** and **`dest_node`** in the **`if __name__ == "__main__":`** block near the end of the file (inside the branch that has both bus and LRT nodes; look for `=== Testing Shortest Path`—defaults are on the order of lines **1542–1543**). Use valid **`node_id`** values from the built network (`data/node.csv`).

### Key files

| File / folder | Purpose |
|---------------|---------|
| `config.py` | Both GTFS paths, ION CSVs, radii, mode constants, wait times. |
| `preprocess_lrt.py` | LRT + extension merge. |
| `build_ion_network.py` | Main multimodal pipeline. |
| `shortest_path_ion.py` | Re-exports `transit.shortest_path`. |
| `data/`, `arcgis_export/`, `aequilibrae_project/` | Outputs as above. |

---

## 5. `transit/shortest_path.py` — shared transit routing

**Where it sits in the flow:** After `bus_network` or `ion_network` has written **`node.csv`** / **`link.csv`**, this module runs **Dijkstra-style** shortest paths on that link table, computes **generalized cost** and path details, and can export a path to GeoPackage. **`zone_to_zone_routing.py`** imports the same cost constants (`COST_GENERALIZED`, `COST_TIME`, `COST_DISTANCE`) and **`shortest_path_transit`** for zone-level OD.

---

## 6. Zone-to-zone routing (`zone_to_zone_routing.py`)

**Pipeline:** (1) Load a **built** network (bus, ion, or road GeoPackage). (2) For a pair of **`zone_id`** values, list all **origin nodes** in the first zone and **destination nodes** in the second. (3) Run node-to-node shortest path for **every** OD pair in that grid (or the relevant API). (4) Return the **best** path by the chosen objective (generalized / time / distance for transit; link column for road).

- **Transit:** `shortest_path_transit` with cost mode from CLI / `zone_to_zone_config.py`.
- **Road:** `road_network/arcgis_export/road_network.gpkg` → NetworkX on **`generalized_cost`** (or another weight column).

**Origin and destination zones** — Defaults live in **`zone-to-zone-shortest-path/zone_to_zone_config.py`**: **`ORIG_ZONE_ID`** and **`DEST_ZONE_ID`** (TAZ / `zone_id` labels on your node tables). Other settings there include **`NETWORK`** (`"bus"` \| `"ion"` \| `"road"`), **`TRANSIT_COST`**, and optional CSV/GPKG overrides. **Command-line flags override the file**, e.g. `--orig 22001 --dest 21414`.

```bash
python zone-to-zone-shortest-path/zone_to_zone_routing.py
python zone-to-zone-shortest-path/zone_to_zone_routing.py --network ion --orig 22001 --dest 21414
python zone-to-zone-shortest-path/zone_to_zone_routing.py --network road --weight-field generalized_cost
python zone-to-zone-shortest-path/zone_to_zone_routing.py --help
```

**Prerequisites:** Built network files exist and nodes include **`zone_id`**.

**Full zone-to-zone matrices (road + bus + ion, generalized cost)** — Use `zone-to-zone-shortest-path/zone_to_zone_matrix.py` to compute **all OD zone pairs** from your zone file (default: `Data/ion_buffer_zones.xlsx`; it uses **`ID_TAZ`** as the zone column by default). For each `(zone_i, zone_j)` and each network, it runs node-to-node shortest paths across all node pairs between those zones and keeps the best generalized-cost path. Outputs are written to `zone-to-zone-shortest-path/`:
- `zone_to_zone_generalized_cost_road_matrix.csv`
- `zone_to_zone_generalized_cost_bus_matrix.csv`
- `zone_to_zone_generalized_cost_ion_matrix.csv`
- `zone_to_zone_generalized_cost_all_networks_long.csv` (combined long-format table with `network` column)

```bash
python zone-to-zone-shortest-path/zone_to_zone_matrix.py
python zone-to-zone-shortest-path/zone_to_zone_matrix.py --network road
python zone-to-zone-shortest-path/zone_to_zone_matrix.py --network bus
python zone-to-zone-shortest-path/zone_to_zone_matrix.py --network ion
```

**One-run analysis + visualization** — After `zone_to_zone_generalized_cost_all_networks_long.csv` is created, run:

```bash
python zone-to-zone-shortest-path/analyze_mode_costs.py
```

This creates:
- Summary tables in `zone-to-zone-shortest-path/analysis/`
- Figures (PNG) in `zone-to-zone-shortest-path/analysis/figures/`, including:
  - generalized cost distribution by network
  - generalized cost vs distance (by mode)
  - generalized cost vs in-vehicle time (by mode)
  - generalized cost vs transfers (bus/ion)
  - binned distance comparison (road vs bus vs ion)

**Map-ready mode-difference layers (ArcGIS Pro / QGIS)** — Export zone-level and OD-line difference layers:

```bash
python zone-to-zone-shortest-path/export_mode_difference_maps.py
```

Default output:
- `zone-to-zone-shortest-path/analysis/mode_cost_difference_maps.gpkg`

GeoPackage layers (zones only, filtered to analyzed origin zones per comparison):
- `zone_delta_bus_minus_road`
- `zone_delta_ion_minus_road`
- `zone_delta_bus_minus_ion`

---

## 7. `traffic_assignment_problem/` — road TAP (AequilibraE)

**Pipeline:** (1) Open **`road_network/aequilibrae_project`**. (2) Choose **one centroid node per TAZ** in the study subset (`road_assignment.py` + GMNS `node.csv` with `zone_id`). (3) Build an OD matrix from **`od_matrix.csv`** (long format: `zone_id_from`, `zone_id_to`, `demand`). (4) Run **`TrafficAssignment`** (e.g. BPR, Frank–Wolfe family). (5) Optionally export **`tap_assignment_road.gpkg`** and link-level CSV (`tap.py`, `assignment_export.py`).

| File | Purpose |
|------|---------|
| `road_assignment.py` | Project helpers, centroids, OD matrix, `build_traffic_assignment`, graph patch. |
| `tap.py` | CLI for the TAP pipeline above. |
| `assignment_export.py` | GeoPackage export of assignment results (+ optional path layers). |
| `od_matrix.csv` | OD table; must match **`TAZ_SUBSET_EIGHT`** (or your configured subset) in `road_assignment.py`. |

```bash
python traffic_assignment_problem/tap.py
python traffic_assignment_problem/tap.py --list-modes
python traffic_assignment_problem/tap.py --export-results-csv
```

**Eight TAZs and OD demand** — The study zones are fixed in **`traffic_assignment_problem/road_assignment.py`** as the tuple **`TAZ_SUBSET_EIGHT`** (eight integer TAZ codes in **row/column order** for the 8×8-style logic). **`tap.py`** imports that tuple to pick **one centroid `node_id` per TAZ** from `road_network/data/gmns/node.csv` and to know which TAZ codes are valid in **`od_matrix.csv`**. To use a **different** set of zones: (1) edit **`TAZ_SUBSET_EIGHT`** in `road_assignment.py` (keep order consistent with how you want centroids and matrix rows aligned); (2) update **`traffic_assignment_problem/od_matrix.csv`** so `zone_id_from` / `zone_id_to` only use TAZs in that subset (and include the trips you need). Optionally pass another OD file with **`python traffic_assignment_problem/tap.py --od-matrix path/to.csv`**.

Ensure the road project has **time + capacity** on links for BPR (see `road_network/config.py`).

---

## Dependency overview

Install Python dependencies from the repo root:

```bash
pip install -r requirements.txt
```

**`requirements.txt`** lists the packages used by the main workflows: **NumPy**, **pandas**, **GeoPandas**, **Shapely**, **NetworkX**, **SciPy** (node clustering on roads), and **AequilibraE** (TAP and project databases). GeoPandas will pull typical geospatial stacks (e.g. **pyogrio** / **fiona** depending on version). Use a **Python version** compatible with your **AequilibraE** build (see [AequilibraE Python documentation](https://www.aequilibrae.com/latest/python/)).

If `pip install` fails for a geospatial wheel on your machine, install a **GeoPandas** stack from **conda-forge** instead (e.g. `conda install -c conda-forge geopandas`) and then `pip install aequilibrae` for the remaining packages.

---

## Suggested order of operations

1. Place or verify inputs under **`Data/`** (roads, lights, TAZ, GTFS, ION CSVs).  
2. Build **road** → `python road_network/build_road_network.py`.  
3. Build **bus** and/or **ION** as needed → `bus_network/build_bus_network.py`, `ion_network/build_ion_network.py`.  
4. **Zone-to-zone** → edit `zone-to-zone-shortest-path/zone_to_zone_config.py`, then `python zone-to-zone-shortest-path/zone_to_zone_routing.py`.  
5. **TAP** → `python traffic_assignment_problem/tap.py`.

---

## Visualizing outputs in ArcGIS Pro

Python builds write **GeoPackage (`.gpkg`)** files under each package’s **`arcgis_export/`** folder (and TAP writes under **`traffic_assignment_problem/arcgis_export/`**). Examples: `road_network/arcgis_export/road_network.gpkg`, `bus_network/arcgis_export/bus_network.gpkg`, `ion_network/arcgis_export/ion_network.gpkg`, `traffic_assignment_problem/arcgis_export/tap_assignment_road.gpkg`. Some tools also write a small **`shortest_path.gpkg`** next to the network export when you run a path test. After those files exist on disk, you can map them in ArcGIS Pro as follows.

1. **Open or create an ArcGIS Pro project** (`*.aprx`)—a blank map is enough.

2. **Add the GeoPackage** — On the **Map** tab, use **Add Data** (or the Catalog pane), browse to the **`.gpkg` file**, and add it to the map. A single GPKG can hold **more than one layer**; pick the ones you need.

3. **Layers you will see** — Network exports usually include **`links`** and often **`nodes`** (line and point geometry). A **shortest-path** export may be **links (and sometimes nodes) only** for the path. **TAP** assignment exports typically include **`assignment_links`** (join of network links to volumes/times); optional runs add **`path_links`** / **`path_nodes`** for sample OD paths.

4. **Filter what is drawn** — Select the layer in the **Contents** pane → open the **layer properties** (or use the **Filter** / **Definition Query** tab, depending on your Pro version). Build a **definition query** (e.g. `demand_tot > 0` on assignment links, or a `link_type` / `zone_id` filter) so only the features you care about draw. You can also set **symbology** (graduated colors, width by field) to emphasize volumes, time, or cost.

5. **Coordinate system** — Layers are stored in **WGS 84** (`EPSG:4326`) for these exports unless you changed the build; set the **map’s coordinate system** if you need to match local basemaps or CAD grids.

Repeat **Add Data** for each GPKG you want to compare (e.g. base network vs TAP result) and align definition queries/symbology for a consistent visual story.


# Analysis
Creating buffer zone in ArcGIS using the Buffer function


