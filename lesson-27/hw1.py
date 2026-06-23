import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# ── Dataset ───────────────────────────────────────────────────────────────────
X_raw, _ = make_moons(n_samples=300, noise=0.08, random_state=42)
scaler = StandardScaler()
X = scaler.fit_transform(X_raw)

# ── Hyperparameter Configs ────────────────────────────────────────────────────
configs = [
    {"eps": 0.5,  "min_samples": 3,  "label": "eps=0.5, min=3"},
    {"eps": 0.3,  "min_samples": 5,  "label": "eps=0.3, min=5"},
    {"eps": 0.2,  "min_samples": 5,  "label": "eps=0.2, min=5"},
    {"eps": 0.3,  "min_samples": 10, "label": "eps=0.3, min=10"},
    {"eps": 0.15, "min_samples": 5,  "label": "eps=0.15, min=5"},
    {"eps": 0.8,  "min_samples": 5,  "label": "eps=0.8, min=5"},
]

# ── Fit & Print ───────────────────────────────────────────────────────────────
results = []
print(f"{'Config':<25} | {'Clusters':>8} | {'Noise':>6} | {'Labels (first 10)'}")
print("-" * 70)
for cfg in configs:
    db = DBSCAN(eps=cfg["eps"], min_samples=cfg["min_samples"])
    db.fit(X)
    n_clusters = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
    n_noise    = int(np.sum(db.labels_ == -1))
    results.append({**cfg, "labels": db.labels_, "n_clusters": n_clusters, "n_noise": n_noise})
    print(f"{cfg['label']:<25} | {n_clusters:>8} | {n_noise:>6} | {db.labels_[:10]}")

best = results[1]
print(f"\n✅ Best config → {best['label']}")
print(f"   n_clusters_ = {best['n_clusters']}")
print(f"   noise pts   = {best['n_noise']}")
print(f"   labels_     = {best['labels']}")

# ── KMeans Comparison ─────────────────────────────────────────────────────────
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
km_labels = kmeans.fit_predict(X)

# ── Visualization ─────────────────────────────────────────────────────────────
PALETTE    = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]
NOISE_CLR  = "#111111"

fig = plt.figure(figsize=(18, 14))
fig.patch.set_facecolor("#F8F9FA")
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

for i, res in enumerate(results):
    row, col = divmod(i, 3)
    ax = fig.add_subplot(gs[row, col])
    ax.set_facecolor("#FFFFFF")

    labels = res["labels"]
    unique = [l for l in sorted(set(labels)) if l != -1]

    noise_mask = labels == -1
    if noise_mask.any():
        ax.scatter(X[noise_mask, 0], X[noise_mask, 1],
                   c=NOISE_CLR, s=18, alpha=0.6, label="Noise", zorder=2)

    for j, lbl in enumerate(unique):
        mask = labels == lbl
        ax.scatter(X[mask, 0], X[mask, 1],
                   c=PALETTE[j % len(PALETTE)],
                   s=22, alpha=0.75, label=f"C{lbl}", zorder=3)

    ax.set_title(
        f"{res['label']}\nclusters={res['n_clusters']}  noise={res['n_noise']}",
        fontsize=9.5, fontweight="bold", pad=6
    )
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor("#CCCCCC")

# KMeans panel (bottom-right)
ax_km = fig.add_subplot(gs[2, 2])
ax_km.set_facecolor("#FFFFFF")
for k in range(2):
    mask = km_labels == k
    ax_km.scatter(X[mask, 0], X[mask, 1],
                  c=PALETTE[k], s=22, alpha=0.75, zorder=3)
centers = kmeans.cluster_centers_
ax_km.scatter(centers[:, 0], centers[:, 1],
              c="black", marker="X", s=160, zorder=5, label="Centroids")
ax_km.set_title("KMeans (k=2) — comparison\nclusters=2  noise=0",
                fontsize=9.5, fontweight="bold", pad=6)
ax_km.set_xticks([]); ax_km.set_yticks([])
for spine in ax_km.spines.values():
    spine.set_edgecolor("#CCCCCC")

fig.suptitle("DBSCAN Hyperparameter Tuning  ·  make_moons dataset",
             fontsize=15, fontweight="bold", y=0.98, color="#222222")

plt.savefig("dbscan_analysis.png", dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.show()
print("\nPlot saved as dbscan_analysis.png")