import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

class KMeans:
    def __init__(self, n_clusters, max_iter=300, tol=1e-4, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None
        self.n_iter_ = None

    def _init_centroids(self, X):
        rng = np.random.RandomState(self.random_state)
        idx = rng.choice(len(X), self.n_clusters, replace=False)
        return X[idx].copy()

    def _assign_labels(self, X, centroids):
        diffs = X[:, np.newaxis, :] - centroids[np.newaxis, :, :]
        dists = np.linalg.norm(diffs, axis=2)
        return np.argmin(dists, axis=1)

    def _compute_inertia(self, X, labels, centroids):
        total = 0.0
        for k in range(self.n_clusters):
            mask = labels == k
            if mask.any():
                total += np.sum((X[mask] - centroids[k]) ** 2)
        return total

    def fit(self, X):
        centroids = self._init_centroids(X)
        for i in range(1, self.max_iter + 1):
            labels = self._assign_labels(X, centroids)
            new_centroids = np.array([
                X[labels == k].mean(axis=0) if (labels == k).any() else centroids[k]
                for k in range(self.n_clusters)
            ])
            shift = np.linalg.norm(new_centroids - centroids)
            centroids = new_centroids
            if shift < self.tol:
                break
        self.cluster_centers_ = centroids
        self.labels_ = self._assign_labels(X, centroids)
        self.inertia_ = self._compute_inertia(X, self.labels_, centroids)
        self.n_iter_ = i
        return self

    def predict(self, X):
        return self._assign_labels(X, self.cluster_centers_)

    def fit_predict(self, X):
        self.fit(X)
        return self.labels_


# ── Dataset ───────────────────────────────────────────────────────────────────
X, y_true = make_blobs(n_samples=300, centers=3, cluster_std=0.8, random_state=42)

# ── Fit ───────────────────────────────────────────────────────────────────────
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)

print("cluster_centers_:\n", np.round(kmeans.cluster_centers_, 4))
print("\ninertia_ :", round(kmeans.inertia_, 4))
print("n_iter_  :", kmeans.n_iter_)

# ── Plot ──────────────────────────────────────────────────────────────────────
X2 = X[:, :2]
centers2 = kmeans.cluster_centers_[:, :2]
colors = ["#4C72B0", "#DD8452", "#55A868"]

fig, ax = plt.subplots(figsize=(7, 5))
for k in range(kmeans.n_clusters):
    mask = kmeans.labels_ == k
    ax.scatter(X2[mask, 0], X2[mask, 1],
               color=colors[k], alpha=0.6, s=40, label=f"Cluster {k}")

ax.scatter(centers2[:, 0], centers2[:, 1],
           c="black", marker="X", s=200, zorder=5, label="Centroids")

ax.set_title("KMeans Clustering (make_blobs)", fontsize=13, fontweight="bold")
ax.set_xlabel("Feature 1")
ax.set_ylabel("Feature 2")
ax.legend()
plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/kmeans_plot.png", dpi=150)
plt.show()
print("Plot saved.")