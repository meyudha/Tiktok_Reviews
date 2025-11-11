"""
Faster/lightweight clustering for ./cleaned_reviews/tiktok_reviews_cleaned.csv
Uses MiniBatchKMeans + sampled silhouette to avoid O(n^2) blowup.
Also runs DBSCAN but keeps it lightweight.

Key ideas:
- silhouette_score(..., sample_size=...) for speed
- MiniBatchKMeans instead of full KMeans
- PCA(2) for visualization
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 5)
RANDOM_STATE = 42

DATA_PATH = "./cleaned_reviews/tiktok_reviews_cleaned.csv"
K_MIN, K_MAX = 2, 6                # keep small to be fast
SIL_SAMPLE_MAX = 10000             # silhouette sample size cap
MBK_BATCH = 256                    # mini-batch size (tune if needed)
DBSCAN_EPS = 0.8
DBSCAN_MIN_SAMPLES = 10

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"File tidak ditemukan: {DATA_PATH}")

df = pd.read_csv(DATA_PATH, low_memory=False)
print("Loaded:", DATA_PATH, "rows:", len(df))

# use numeric features only (you reported 4: ['score','thumbs_up','thumbs_up_capped','content_len'])
num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
if not num_cols:
    raise SystemExit("Tidak ada fitur numerik untuk clustering.")
print("Numeric features:", num_cols)

X = df[num_cols].dropna().copy()
n_samples = X.shape[0]
print("Using rows:", n_samples)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

sil_sample = min(SIL_SAMPLE_MAX, n_samples)
print(f"Silhouette sample size: {sil_sample}")

ks = list(range(K_MIN, K_MAX + 1))
inertias = []
sil_scores = []

print("\nRunning MiniBatchKMeans over k =", ks)
for k in ks:
    mbk = MiniBatchKMeans(n_clusters=k, random_state=RANDOM_STATE, batch_size=MBK_BATCH, n_init=10)
    labels = mbk.fit_predict(X_scaled)
    inertias.append(float(mbk.inertia_))
    # compute silhouette on a sample to speed up
    try:
        if n_samples > sil_sample:
            # silhouette_score handles sampling internally only if sample_size given (sklearn >=0.24)
            sil = silhouette_score(X_scaled, labels, sample_size=sil_sample, random_state=RANDOM_STATE)
        else:
            sil = silhouette_score(X_scaled, labels)
    except Exception as e:
        print(f"  Warning: silhouette_score failed for k={k} ({e}). Setting silhouette=nan")
        sil = float("nan")
    sil_scores.append(float(sil))
    print(f"  k={k}: inertia={inertias[-1]:.2f}, sil={sil_scores[-1]}")

plt.figure()
plt.plot(ks, inertias, marker='o')
plt.xlabel("k")
plt.ylabel("Inertia")
plt.title("Elbow (MiniBatchKMeans)")
plt.show()

plt.figure()
plt.plot(ks, sil_scores, marker='o', color='darkorange')
plt.xlabel("k")
plt.ylabel("Silhouette (sampled)")
plt.title("Sampled Silhouette Score")
plt.show()

# prefer max silhouette if available, else elbow heuristic (largest drop in inertia)
if np.all(np.isnan(sil_scores)):
    # fallback to elbow heuristic: largest relative drop in inertia
    diffs = np.diff(inertias)
    if len(diffs) >= 2:
        ratio = np.abs(diffs[1:] / (diffs[:-1] + 1e-9))
        elbow_idx = int(np.argmin(ratio)) + K_MIN + 1
        optimal_k = elbow_idx
    else:
        optimal_k = ks[0]
    print(f"No valid silhouette scores; fallback optimal_k (elbow heuristic) = {optimal_k}")
else:
    # prefer k with highest silhouette (break ties by higher inertia improvement)
    optimal_k = ks[int(np.nanargmax(sil_scores))]
    print(f"Optimal k chosen by sampled silhouette = {optimal_k}")

print(f"\nFitting final MiniBatchKMeans with k={optimal_k} ...")
final_km = MiniBatchKMeans(n_clusters=optimal_k, random_state=RANDOM_STATE, batch_size=MBK_BATCH, n_init=20)
labels_km = final_km.fit_predict(X_scaled)
df.loc[X.index, 'Cluster_KMeans'] = labels_km

# compute final silhouette on sample (if possible)
try:
    if n_samples > sil_sample:
        sil_final = silhouette_score(X_scaled, labels_km, sample_size=sil_sample, random_state=RANDOM_STATE)
    else:
        sil_final = silhouette_score(X_scaled, labels_km)
except Exception:
    sil_final = float("nan")
print(f"K-Means final silhouette (sampled): {sil_final}")

print("\nRunning DBSCAN (may be slow if n large) ...")
# If data too large, subsample for DBSCAN or tune eps by domain knowledge
if n_samples > 20000:
    print("Large dataset detected — DBSCAN will run on a 20k subsample for speed.")
    sample_idx = np.random.RandomState(RANDOM_STATE).choice(np.arange(n_samples), size=20000, replace=False)
    X_for_db = X_scaled[sample_idx]
    db = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES, n_jobs=-1)
    labels_db_sub = db.fit_predict(X_for_db)
    # Map labels back loosely: assign label -1 for points not in sample
    full_labels_db = np.full(n_samples, -1, dtype=int)
    full_labels_db[sample_idx] = labels_db_sub
    df.loc[X.index, 'Cluster_DBSCAN'] = full_labels_db
    n_clusters_dbscan = len(set(labels_db_sub)) - (1 if -1 in labels_db_sub else 0)
else:
    db = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES, n_jobs=-1)
    labels_db = db.fit_predict(X_scaled)
    df.loc[X.index, 'Cluster_DBSCAN'] = labels_db
    n_clusters_dbscan = len(set(labels_db)) - (1 if -1 in labels_db else 0)

print(f"DBSCAN clusters (excluding noise): {n_clusters_dbscan}")

print("\nPCA 2D projection for visualization ...")
pca = PCA(n_components=2, random_state=RANDOM_STATE)
X_pca = pca.fit_transform(X_scaled)
df.loc[X.index, 'pca1'] = X_pca[:, 0]
df.loc[X.index, 'pca2'] = X_pca[:, 1]

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
sns.scatterplot(x='pca1', y='pca2', hue='Cluster_KMeans', data=df.loc[X.index], palette='tab10', s=30, alpha=0.8)
plt.title(f"MiniBatchKMeans k={optimal_k} (PCA 2D)")

plt.subplot(1,2,2)
sns.scatterplot(x='pca1', y='pca2', hue='Cluster_DBSCAN', data=df.loc[X.index], palette='tab10', s=30, alpha=0.8)
plt.title("DBSCAN (PCA 2D)")

plt.tight_layout()
plt.show()

print("\nSummary:")
print(f" - n_samples = {n_samples}")
print(f" - K candidates = {ks}")
print(f" - inertias = {inertias}")
print(f" - sampled silhouette scores = {sil_scores}")
print(f" - selected optimal_k = {optimal_k}")
print(f" - final KMeans silhouette (sampled) = {sil_final}")
print(f" - DBSCAN clusters (excl noise) = {n_clusters_dbscan}")

centroids = final_km.cluster_centers_
centroids_unscaled = scaler.inverse_transform(centroids)
centroid_df = pd.DataFrame(centroids_unscaled, columns=num_cols)
print("\nCluster centroids (approx, original scale):")
print(centroid_df.round(3))

out_csv = "./clustered_tiktok_reviews_fast.csv"
df.to_csv(out_csv, index=False)
print("\nSaved clustered output to:", out_csv)
