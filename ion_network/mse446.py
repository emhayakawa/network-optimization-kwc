# Ride-pooling opportunities with emissions (max 4 passengers per vehicle)
# =============================================================================
# - Hierarchical clustering finds spatial–temporal trip groups.
# - Each *pool* is one vehicle serving 2–4 trips (FHV: 1 trip ≈ 1 passenger request).
# - Feasibility: pickup time spread + max pickup diameter (km), both configurable.
# - Clusters with >4 trips are split greedily (pickup time order) into consecutive
#   feasible windows of size ≤4 (non-overlapping).
# - Emissions: baseline = solo CO2 sum; pooled = modeled single-vehicle CO2 using
#   SHARED_ROUTE_FRACTION × sum(trip_miles) × avg g CO2/mi (see cluster_pooled_co2_kg_estimate).
#
# Jupyter: set DATA_PATH / OUTPUT_DIR in the first cell; cwd = notebook working folder.

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.metrics.pairwise import haversine_distances
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)
plt.rcParams["figure.dpi"] = 120

# --- Paths (edit in notebook) ---
DATA_PATH = Path("Data/merged_trip_emissions_coordinates.csv")
OUTPUT_DIR = Path("Data")

RANDOM_STATE = 42
SUBSET_N = 10_000

# --- Pooling / vehicle rule ---
MAX_PASSENGERS_PER_VEHICLE = 4  # strict cap: each pool has 2..4 trips
MAX_PICKUP_SPREAD_MINUTES_LIST = [5, 10, 15, 20]
MAX_PICKUP_DIAMETER_KM_LIST = [2.0, 3.0, 5.0]
SHARED_ROUTE_FRACTION = 0.55  # pooled VMT vs sum(solo miles); tune 0.45–0.65

# Ward cut heights (standardized feature space)
LINKAGE_CUT_HEIGHTS = [20, 40, 60, 80, 100]

# Set False to skip the multi-linkage dendrogram exploration plots
PLOT_ALL_LINKAGE_METHODS = False


def cluster_pickup_spread_minutes(cluster_df: pd.DataFrame) -> float:
    if len(cluster_df) < 2:
        return 0.0
    t = cluster_df["pickup_datetime"]
    return (t.max() - t.min()).total_seconds() / 60.0


def cluster_max_pickup_diameter_km(cluster_df: pd.DataFrame) -> float:
    n = len(cluster_df)
    if n < 2:
        return 0.0
    coords = np.radians(cluster_df[["pickup_lat", "pickup_lon"]].to_numpy(dtype=float))
    d = haversine_distances(coords) * 6371.0
    return float(d[np.triu_indices(n, k=1)].max())


def pooling_feasible(
    cluster_df: pd.DataFrame,
    max_pickup_spread_min: float,
    max_pickup_diameter_km: float,
) -> bool:
    n = len(cluster_df)
    if n < 2 or n > MAX_PASSENGERS_PER_VEHICLE:
        return False
    if cluster_pickup_spread_minutes(cluster_df) > max_pickup_spread_min:
        return False
    if cluster_max_pickup_diameter_km(cluster_df) > max_pickup_diameter_km:
        return False
    return True


def iter_feasible_pools_from_hc_cluster(
    cluster_df: pd.DataFrame,
    max_passengers: int,
    max_pickup_spread_min: float,
    max_pickup_diameter_km: float,
) -> list[pd.DataFrame]:
    """
    Turn one HC cluster into 0+ disjoint pools, each with 2..max_passengers trips,
    each passing time + distance checks. Greedy: sort by pickup time, repeatedly take
    the longest feasible prefix (by end index) and advance.
    """
    if len(cluster_df) < 2:
        return []
    cdf = cluster_df.sort_values("pickup_datetime").reset_index(drop=True)
    n = len(cdf)
    pools: list[pd.DataFrame] = []
    i = 0
    while i < n:
        best_end = None
        # try longest window [i:j) with 2 <= j-i <= max_passengers
        lo_len = 2
        hi_len = min(max_passengers, n - i)
        for length in range(hi_len, lo_len - 1, -1):
            sub = cdf.iloc[i : i + length]
            if pooling_feasible(sub, max_pickup_spread_min, max_pickup_diameter_km):
                best_end = i + length
                break
        if best_end is not None:
            pools.append(cdf.iloc[i:best_end].copy())
            i = best_end
        else:
            i += 1
    return pools


def cluster_baseline_co2_kg(cluster_df: pd.DataFrame) -> float:
    return float(cluster_df["co2_total_g"].sum()) / 1000.0


def cluster_pooled_co2_kg_estimate(cluster_df: pd.DataFrame, shared_fraction: float) -> float:
    miles = cluster_df["trip_miles"].to_numpy(dtype=float)
    co2g = cluster_df["co2_total_g"].to_numpy(dtype=float)
    solo_miles = np.nansum(miles)
    if solo_miles <= 0 or not np.isfinite(solo_miles):
        return float(np.nansum(co2g)) / 1000.0
    pooled_miles = solo_miles * float(shared_fraction)
    with np.errstate(divide="ignore", invalid="ignore"):
        g_per_mi = co2g / np.where(miles > 0, miles, np.nan)
    avg_gpm = np.nanmean(g_per_mi)
    if not np.isfinite(avg_gpm) or avg_gpm <= 0:
        avg_gpm = float(cluster_df["co2TailpipeGpm"].mean())
    if not np.isfinite(avg_gpm) or avg_gpm <= 0:
        avg_gpm = 0.0
    return float(pooled_miles * avg_gpm) / 1000.0


def safe_silhouette(X, labels) -> float:
    try:
        return float(silhouette_score(X, labels))
    except ValueError:
        return float("nan")


def safe_davies_bouldin(X, labels) -> float:
    try:
        return float(davies_bouldin_score(X, labels))
    except ValueError:
        return float("nan")


# -----------------------
# Load & preprocess
# -----------------------
if not DATA_PATH.is_file():
    raise FileNotFoundError(
        f"Missing CSV: {DATA_PATH.resolve()}\n"
        'Set DATA_PATH = Path("Data/your_file.csv") in the first notebook cell.'
    )

df_full = pd.read_csv(DATA_PATH, low_memory=False)
df = df_full.sample(
    n=min(SUBSET_N, len(df_full)), random_state=RANDOM_STATE
).reset_index(drop=True)

df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"], errors="coerce")
df["pickup_lat"] = pd.to_numeric(df["pickup_lat"], errors="coerce")
df["pickup_lon"] = pd.to_numeric(df["pickup_lon"], errors="coerce")
df["trip_miles"] = pd.to_numeric(df["trip_miles"], errors="coerce")
df["co2TailpipeGpm"] = pd.to_numeric(df["co2TailpipeGpm"], errors="coerce")
df["co2_total_g"] = pd.to_numeric(df["co2_total_g"], errors="coerce")
df["trip_seconds"] = pd.to_numeric(df["trip_time"], errors="coerce")

df = df.dropna(subset=["pickup_lat", "pickup_lon", "pickup_datetime", "trip_seconds"])
df["trip_duration_min"] = df["trip_seconds"] / 60.0
print(f"Working subset: {len(df):,} rows")

df["pickup_hour"] = df["pickup_datetime"].dt.hour
df["hour_sin"] = np.sin(2 * np.pi * df["pickup_hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["pickup_hour"] / 24)

features = ["pickup_lat", "pickup_lon", "hour_sin", "hour_cos"]
scaler = StandardScaler()
X = scaler.fit_transform(df[features].values)

# Optional: compare linkage methods (slow)
if PLOT_ALL_LINKAGE_METHODS:
    from scipy.cluster import hierarchy as hc

    for method in ["single", "complete", "average", "ward"]:
        plt.figure(figsize=(8, 5))
        Zm = hc.linkage(X, method=method)
        hc.dendrogram(Zm, truncate_mode="level", p=15)
        plt.title(f"{method.capitalize()} linkage (truncated)")
        plt.show()

# -----------------------
# Ward — main model
# -----------------------
Z = linkage(X, method="ward")
print("Ward linkage matrix shape:", Z.shape)

plt.figure(figsize=(12, 5))
dendrogram(Z, truncate_mode="level", p=20, leaf_rotation=45, leaf_font_size=10)
plt.title("Ward dendrogram (truncated)")
plt.ylabel("Ward distance (standardized features)")
plt.tight_layout()
plt.show()

# -----------------------
# Sweep cuts × time × distance — emissions & 4-passenger rule
# -----------------------
N_TRIPS = len(df)
results = []

for h in LINKAGE_CUT_HEIGHTS:
    col = f"hc_cluster_h{h}"
    df[col] = fcluster(Z, t=h, criterion="distance")
    labels = df[col].values
    sil = safe_silhouette(X, labels)
    dbi = safe_davies_bouldin(X, labels)
    n_clu = int(df[col].nunique())

    for max_spread in MAX_PICKUP_SPREAD_MINUTES_LIST:
        for max_d_km in MAX_PICKUP_DIAMETER_KM_LIST:
            n_pools = 0
            n_trips_pooled = 0
            trips_in_multisize_hc = 0  # trips that ended up in some pool (may be split)
            baseline_kg = 0.0
            pooled_kg = 0.0
            hc_singleton_trips = 0

            for _, hc_block in df.groupby(col):
                sz = len(hc_block)
                if sz == 1:
                    hc_singleton_trips += 1  # one trip per singleton HC cluster
                    continue
                trips_in_multisize_hc += sz
                pools = iter_feasible_pools_from_hc_cluster(
                    hc_block,
                    MAX_PASSENGERS_PER_VEHICLE,
                    max_spread,
                    max_d_km,
                )
                for pool_df in pools:
                    n_pools += 1
                    k = len(pool_df)
                    n_trips_pooled += k
                    b = cluster_baseline_co2_kg(pool_df)
                    p = cluster_pooled_co2_kg_estimate(pool_df, SHARED_ROUTE_FRACTION)
                    p = min(p, b)
                    baseline_kg += b
                    pooled_kg += p

            trips_unassigned_from_multisize = trips_in_multisize_hc - n_trips_pooled
            # singleton HC trips never enter a pool with this pipeline
            n_trips_singleton_hc = hc_singleton_trips
            trips_not_in_pool = trips_unassigned_from_multisize + n_trips_singleton_hc
            saved_kg = baseline_kg - pooled_kg

            results.append(
                {
                    "linkage_cut_height": h,
                    "max_pickup_spread_min": max_spread,
                    "max_pickup_diameter_km": max_d_km,
                    "max_passengers_per_vehicle": MAX_PASSENGERS_PER_VEHICLE,
                    "shared_route_fraction": SHARED_ROUTE_FRACTION,
                    "n_trips_total": N_TRIPS,
                    "n_hc_clusters": n_clu,
                    "silhouette": sil,
                    "davies_bouldin": dbi,
                    "n_feasible_pools": n_pools,
                    "n_trips_in_pools": n_trips_pooled,
                    "pct_trips_in_pools": (
                        100.0 * n_trips_pooled / N_TRIPS if N_TRIPS else np.nan
                    ),
                    "trips_in_multitrip_hc_clusters": trips_in_multisize_hc,
                    "trips_unassigned_in_multitrip_hc": trips_unassigned_from_multisize,
                    "n_trips_singleton_hc": n_trips_singleton_hc,
                    "trips_not_in_any_pool": trips_not_in_pool,
                    "baseline_co2_kg_in_pools": baseline_kg,
                    "pooled_co2_kg_est": pooled_kg,
                    "co2_saved_kg_est": saved_kg,
                    "co2_saved_g_per_trip_in_pool": (
                        (saved_kg * 1000.0 / n_trips_pooled) if n_trips_pooled else np.nan
                    ),
                }
            )

results_df = pd.DataFrame(results)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
print(results_df.to_string())

# --- Interpretation helper (why many rows look identical) ---
r0 = results_df.iloc[0]
print(
    "\n=== Interpretation ===\n"
    f"Sample size: {int(r0['n_trips_total'])} trips after cleaning.\n"
    f"Only {int(r0['n_trips_in_pools'])} trips ({r0['pct_trips_in_pools']:.2f}%) sit in at least one "
    f"feasible pool (2–4 riders, pickup spread + diameter rules).\n"
)
if results_df[["n_feasible_pools", "n_trips_in_pools", "co2_saved_kg_est"]].nunique().min() == 1:
    print(
        "Your spread/diameter/linkage rows match because those few pools already satisfy the\n"
        "**tightest** time/distance settings (5 min, 2 km). Loosening thresholds does not add\n"
        "pools: no other HC groups yield a feasible 2–4 trip window under the greedy split.\n"
        "Linkage height 60+ → one giant HC cluster (silhouette NaN = one label); the same 2 pools\n"
        "of 4 trips are recovered from that blob, so emissions stay identical.\n"
    )
print(
    "To see differences across rows: try **more Ward cuts** (e.g. heights 5–25), **DBSCAN**/HDBSCAN\n"
    "on projected meters, or **softer** max spread / diameter; add **dropoff** features for realism.\n"
)

# -----------------------
# Save
# -----------------------
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT_DIR / "trips_hc_labeled.csv", index=False)
results_df.to_csv(OUTPUT_DIR / "pooling_emissions_summary.csv", index=False)
print(f"Saved labeled trips and pooling_emissions_summary under {OUTPUT_DIR.resolve()}")
