"""
Body Fat Dataset Simulation: PCA→KMeans vs KMeans→PCA
Comparing dimensionality reduction timing with Euclidean and Cosine distances
Author: Xinyue
Date: October 2024
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Set random seed
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print("="*70)
print("BODY FAT SIMULATION: PCA→KMEANS VS KMEANS→PCA")
print("="*70)

# ============================================================================
# SECTION 1: LOAD AND PREPROCESS DATA
# ============================================================================

DATA_FILE = "bodyfat.csv"
df = pd.read_csv(DATA_FILE)

print(f"\n[1] Dataset loaded: {df.shape}")

# Select numeric features
X = df.select_dtypes(include=[np.number]).copy()
X = X.loc[:, X.nunique(dropna=False) > 1]

print(f"[2] Numeric features: {X.shape[1]} columns")

# Preprocess
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

Xs = pipeline.fit_transform(X)
print(f"[3] Data preprocessed and scaled: {Xs.shape}")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def l2_normalize_rows(A, eps=1e-12):
    """Normalize each row to unit length (L2 norm)"""
    norms = np.linalg.norm(A, axis=1, keepdims=True)
    return A / np.clip(norms, eps, None)

def discrete_viridis(K):
    """Create discrete viridis colormap for K clusters"""
    cmap = plt.cm.get_cmap("viridis", K)
    norm = mcolors.BoundaryNorm(boundaries=np.arange(-0.5, K + 0.5, 1), ncolors=K)
    ticks = np.arange(K)
    ticklabels = [f"Cluster {i+1}" for i in range(K)]
    return cmap, norm, ticks, ticklabels

def spherical_kmeans(X_unit, K, max_iter=100, tol=1e-4, random_state=42):
    """
    Spherical k-means for cosine distance (unit vectors)
    Uses k-means++ initialization adapted for cosine similarity
    """
    rng = np.random.default_rng(random_state)
    n, d = X_unit.shape
    
    # K-means++ initialization for cosine distance
    first = rng.integers(0, n)
    centroids = [X_unit[first]]
    
    for _ in range(1, K):
        # Cosine similarity to nearest chosen centroid
        sims = np.max(X_unit @ np.vstack(centroids).T, axis=1)
        dist = 1.0 - sims
        probs = np.square(dist)
        probs_sum = probs.sum()
        
        if probs_sum <= 0:
            idx = rng.integers(0, n)
        else:
            probs = probs / probs_sum
            idx = rng.choice(n, p=probs)
        centroids.append(X_unit[idx])
    
    C = np.vstack(centroids)
    C = l2_normalize_rows(C)
    
    labels = np.full(n, -1, dtype=int)
    
    for it in range(max_iter):
        # Assign to nearest by cosine: argmax dot(x, c)
        scores = X_unit @ C.T
        new_labels = np.argmax(scores, axis=1)
        
        # Check convergence
        changes = np.mean(new_labels != labels) if labels[0] != -1 else 1.0
        labels = new_labels
        
        if changes < tol:
            break
        
        # Update centroids
        for k in range(K):
            idx = (labels == k)
            if idx.any():
                C[k] = l2_normalize_rows(X_unit[idx].mean(axis=0, keepdims=True))[0]
            else:
                # Re-seed empty cluster
                sims = np.max(X_unit @ C.T, axis=1)
                far_idx = np.argmin(sims)
                C[k] = X_unit[far_idx]
    
    return labels, C

# ============================================================================
# SECTION 2: EUCLIDEAN DISTANCE COMPARISON
# ============================================================================

print(f"\n{'='*70}")
print("EUCLIDEAN DISTANCE: PCA→KMEANS VS KMEANS→PCA")
print(f"{'='*70}")

K = 2  # Number of clusters (use optimal K from earlier analysis)

# --- (a) PCA BEFORE clustering (Euclidean) ---
print("\n[4] Running: PCA → KMeans (Euclidean)...")

pca_before_eu = PCA(n_components=3, random_state=RANDOM_STATE).fit(Xs)
Z3_before_eu = pca_before_eu.transform(Xs)

km_before_eu = KMeans(n_clusters=K, init='k-means++', n_init=10, random_state=RANDOM_STATE)
labels_before_eu = km_before_eu.fit_predict(Z3_before_eu)
centroids_before_eu = km_before_eu.cluster_centers_

sil_before_eu = silhouette_score(Z3_before_eu, labels_before_eu, sample_size=min(10000, len(Z3_before_eu)))
print(f"    Silhouette Score: {sil_before_eu:.4f}")

# --- (b) PCA AFTER clustering (Euclidean) ---
print("\n[5] Running: KMeans → PCA (Euclidean)...")

km_after_eu = KMeans(n_clusters=K, init='k-means++', n_init=10, random_state=RANDOM_STATE)
labels_after_eu = km_after_eu.fit_predict(Xs)

# Project to 3D for visualization
pca_after_eu = PCA(n_components=3, random_state=RANDOM_STATE).fit(Xs)
Z3_after_eu = pca_after_eu.transform(Xs)
centroids_after_eu = pca_after_eu.transform(km_after_eu.cluster_centers_)

sil_after_eu = silhouette_score(Z3_after_eu, labels_after_eu, sample_size=min(10000, len(Z3_after_eu)))
print(f"    Silhouette Score: {sil_after_eu:.4f}")

# ============================================================================
# SECTION 3: EUCLIDEAN VISUALIZATION
# ============================================================================

print("\n[6] Creating Euclidean comparison visualization...")

cmap, norm, ticks, ticklabels = discrete_viridis(K)

# Optional subsampling for clearer visuals
N_SHOW = min(252, len(Z3_before_eu))  # Show all for Body Fat (small dataset)

from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(14, 6))

# --- Panel (a): PCA → KMeans ---
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
sc1 = ax1.scatter(Z3_before_eu[:, 0], Z3_before_eu[:, 1], Z3_before_eu[:, 2],
                  s=25, alpha=0.7, c=labels_before_eu, cmap=cmap, norm=norm,
                  edgecolors='k', linewidth=0.2)
ax1.scatter(centroids_before_eu[:, 0], centroids_before_eu[:, 1], centroids_before_eu[:, 2],
            s=300, marker='X', edgecolor='black', linewidths=2, facecolor='red', label='Centroids')
ax1.set_title(f"Body Fat - PCA → K-Means++ (Euclidean)\nSilhouette: {sil_before_eu:.4f}", 
              fontsize=11, fontweight='bold')
ax1.set_xlabel("PC 1", labelpad=8)
ax1.set_ylabel("PC 2", labelpad=8)
ax1.set_zlabel("PC 3", labelpad=8)

lim = np.max(np.abs(Z3_before_eu)) * 1.1
ax1.set_xlim(-lim, lim)
ax1.set_ylim(-lim, lim)
ax1.set_zlim(-lim, lim)

cbar1 = fig.colorbar(sc1, ax=ax1, shrink=0.6, pad=0.15)
cbar1.set_ticks(ticks)
cbar1.set_ticklabels(ticklabels)
cbar1.set_label("Cluster")

# --- Panel (b): KMeans → PCA ---
ax2 = fig.add_subplot(1, 2, 2, projection='3d')
sc2 = ax2.scatter(Z3_after_eu[:, 0], Z3_after_eu[:, 1], Z3_after_eu[:, 2],
                  s=25, alpha=0.7, c=labels_after_eu, cmap=cmap, norm=norm,
                  edgecolors='k', linewidth=0.2)
ax2.scatter(centroids_after_eu[:, 0], centroids_after_eu[:, 1], centroids_after_eu[:, 2],
            s=300, marker='X', edgecolor='black', linewidths=2, facecolor='red', label='Centroids')
ax2.set_title(f"Body Fat - K-Means++ → PCA (Euclidean)\nSilhouette: {sil_after_eu:.4f}",
              fontsize=11, fontweight='bold')
ax2.set_xlabel("PC 1", labelpad=8)
ax2.set_ylabel("PC 2", labelpad=8)
ax2.set_zlabel("PC 3", labelpad=8)

ax2.set_xlim(-lim, lim)
ax2.set_ylim(-lim, lim)
ax2.set_zlim(-lim, lim)

cbar2 = fig.colorbar(sc2, ax=ax2, shrink=0.6, pad=0.15)
cbar2.set_ticks(ticks)
cbar2.set_ticklabels(ticklabels)
cbar2.set_label("Cluster")

plt.suptitle("Body Fat: Euclidean Distance - When to Apply PCA?", 
             fontsize=13, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig('bodyfat_simulation_euclidean.png', dpi=300, bbox_inches='tight')
plt.show()

print("    ✓ Euclidean visualization saved")

# ============================================================================
# SECTION 4: COSINE DISTANCE COMPARISON
# ============================================================================

print(f"\n{'='*70}")
print("COSINE DISTANCE: PCA→KMEANS VS KMEANS→PCA")
print(f"{'='*70}")

# --- (a) PCA BEFORE clustering (Cosine) ---
print("\n[7] Running: PCA → Spherical KMeans (Cosine)...")

pca_before_cos = PCA(n_components=3, random_state=RANDOM_STATE).fit(Xs)
Z3_before_cos = pca_before_cos.transform(Xs)
Z3_before_cos_unit = l2_normalize_rows(Z3_before_cos)

labels_before_cos, centroids_before_cos = spherical_kmeans(Z3_before_cos_unit, K, random_state=RANDOM_STATE)

sil_before_cos = silhouette_score(Z3_before_cos, labels_before_cos, metric='cosine',
                                  sample_size=min(10000, len(Z3_before_cos)))
print(f"    Silhouette Score: {sil_before_cos:.4f}")

# --- (b) PCA AFTER clustering (Cosine) ---
print("\n[8] Running: Spherical KMeans → PCA (Cosine)...")

Xs_unit = l2_normalize_rows(Xs)
labels_after_cos, centroids_after_cos = spherical_kmeans(Xs_unit, K, random_state=RANDOM_STATE)

# Project to 3D for visualization
pca_after_cos = PCA(n_components=3, random_state=RANDOM_STATE).fit(Xs_unit)
Z3_after_cos = pca_after_cos.transform(Xs_unit)
centroids_after_cos_pca = pca_after_cos.transform(centroids_after_cos)

sil_after_cos = silhouette_score(Xs_unit, labels_after_cos, metric='cosine',
                                sample_size=min(10000, len(Xs_unit)))
print(f"    Silhouette Score: {sil_after_cos:.4f}")

# ============================================================================
# SECTION 5: COSINE VISUALIZATION
# ============================================================================

print("\n[9] Creating Cosine comparison visualization...")

fig = plt.figure(figsize=(14, 6))

# --- Panel (a): PCA → Spherical KMeans ---
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
sc1 = ax1.scatter(Z3_before_cos[:, 0], Z3_before_cos[:, 1], Z3_before_cos[:, 2],
                  s=25, alpha=0.7, c=labels_before_cos, cmap=cmap, norm=norm,
                  edgecolors='k', linewidth=0.2)
ax1.scatter(centroids_before_cos[:, 0], centroids_before_cos[:, 1], centroids_before_cos[:, 2],
            s=300, marker='X', edgecolor='black', linewidths=2, facecolor='red', label='Centroids')
ax1.set_title(f"Body Fat - PCA → Spherical K-Means (Cosine)\nSilhouette: {sil_before_cos:.4f}",
              fontsize=11, fontweight='bold')
ax1.set_xlabel("PC 1", labelpad=8)
ax1.set_ylabel("PC 2", labelpad=8)
ax1.set_zlabel("PC 3", labelpad=8)

lim_cos = np.max(np.abs(Z3_before_cos)) * 1.1
ax1.set_xlim(-lim_cos, lim_cos)
ax1.set_ylim(-lim_cos, lim_cos)
ax1.set_zlim(-lim_cos, lim_cos)

cbar1 = fig.colorbar(sc1, ax=ax1, shrink=0.6, pad=0.15)
cbar1.set_ticks(ticks)
cbar1.set_ticklabels(ticklabels)
cbar1.set_label("Cluster")

# --- Panel (b): Spherical KMeans → PCA ---
ax2 = fig.add_subplot(1, 2, 2, projection='3d')
sc2 = ax2.scatter(Z3_after_cos[:, 0], Z3_after_cos[:, 1], Z3_after_cos[:, 2],
                  s=25, alpha=0.7, c=labels_after_cos, cmap=cmap, norm=norm,
                  edgecolors='k', linewidth=0.2)
ax2.scatter(centroids_after_cos_pca[:, 0], centroids_after_cos_pca[:, 1], centroids_after_cos_pca[:, 2],
            s=300, marker='X', edgecolor='black', linewidths=2, facecolor='red', label='Centroids')
ax2.set_title(f"Body Fat - Spherical K-Means → PCA (Cosine)\nSilhouette: {sil_after_cos:.4f}",
              fontsize=11, fontweight='bold')
ax2.set_xlabel("PC 1", labelpad=8)
ax2.set_ylabel("PC 2", labelpad=8)
ax2.set_zlabel("PC 3", labelpad=8)

lim_cos2 = np.max(np.abs(Z3_after_cos)) * 1.1
ax2.set_xlim(-lim_cos2, lim_cos2)
ax2.set_ylim(-lim_cos2, lim_cos2)
ax2.set_zlim(-lim_cos2, lim_cos2)

cbar2 = fig.colorbar(sc2, ax=ax2, shrink=0.6, pad=0.15)
cbar2.set_ticks(ticks)
cbar2.set_ticklabels(ticklabels)
cbar2.set_label("Cluster")

plt.suptitle("Body Fat: Cosine Distance - When to Apply PCA?",
             fontsize=13, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig('bodyfat_simulation_cosine.png', dpi=300, bbox_inches='tight')
plt.show()

print("    ✓ Cosine visualization saved")

# ============================================================================
# SECTION 6: SUMMARY
# ============================================================================

print(f"\n{'='*70}")
print("SIMULATION SUMMARY")
print(f"{'='*70}")

summary_data = {
    'Method': [
        'PCA → KMeans (Euclidean)',
        'KMeans → PCA (Euclidean)',
        'PCA → Spherical KMeans (Cosine)',
        'Spherical KMeans → PCA (Cosine)'
    ],
    'Silhouette Score': [
        f"{sil_before_eu:.4f}",
        f"{sil_after_eu:.4f}",
        f"{sil_before_cos:.4f}",
        f"{sil_after_cos:.4f}"
    ]
}

summary_df = pd.DataFrame(summary_data)
print("\n", summary_df.to_string(index=False))

print(f"\n{'='*70}")
print("KEY INSIGHTS:")
print(f"{'='*70}")
print("\n1. EUCLIDEAN DISTANCE:")
print(f"   - PCA→KMeans: {sil_before_eu:.4f}")
print(f"   - KMeans→PCA: {sil_after_eu:.4f}")
if sil_before_eu > sil_after_eu:
    print("   → Better to apply PCA BEFORE clustering")
else:
    print("   → Better to cluster BEFORE applying PCA")

print("\n2. COSINE DISTANCE:")
print(f"   - PCA→KMeans: {sil_before_cos:.4f}")
print(f"   - KMeans→PCA: {sil_after_cos:.4f}")
if sil_before_cos > sil_after_cos:
    print("   → Better to apply PCA BEFORE clustering")
else:
    print("   → Better to cluster BEFORE applying PCA")

print("\n3. RECOMMENDATION:")
print("   For Body Fat dataset, the timing of PCA application")
print("   affects clustering quality. Choose based on your metric priority.")

print(f"\n{'='*70}")
print("✅ SIMULATION COMPLETE!")
print(f"{'='*70}")
print("\nGenerated files:")
print("  - bodyfat_simulation_euclidean.png")
print("  - bodyfat_simulation_cosine.png")
print(f"{'='*70}")