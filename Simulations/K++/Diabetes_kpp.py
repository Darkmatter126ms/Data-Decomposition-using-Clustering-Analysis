"""
K-Means++ Clustering Analysis - Diabetes Dataset
Author: Xinyue
Date: October 2024
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print("="*60)
print("K-MEANS++ CLUSTERING - DIABETES DATASET")
print("="*60)

# ============================================================================
# SECTION 1: DATA LOADING AND PREPROCESSING
# ============================================================================

# Load Diabetes dataset
DATA_FILE = "diabetes_prediction_dataset.csv"
df = pd.read_csv(DATA_FILE)

print(f"\n[1] Dataset loaded successfully")
print(f"    Shape: {df.shape}")
print(f"    Columns: {list(df.columns)}")

# Select numeric features only
X_raw = df.select_dtypes(include=[np.number]).copy()

# Remove constant columns (if any)
X_raw = X_raw.loc[:, X_raw.nunique() > 1]

print(f"\n[2] After selecting numeric columns: {X_raw.shape}")
print(f"    Features: {list(X_raw.columns)}")

# Handle missing values (fill with median)
missing_count = X_raw.isnull().sum().sum()
if missing_count > 0:
    print(f"\n[3] Found {missing_count} missing values - filling with median")
    X_raw = X_raw.fillna(X_raw.median())
else:
    print(f"\n[3] No missing values found")

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

print(f"\n[4] Data standardized (mean=0, std=1)")
print(f"    Final data shape: {X_scaled.shape}")

# ============================================================================
# SECTION 2: DIMENSIONALITY REDUCTION WITH PCA
# ============================================================================

# Apply PCA for 2D visualization
pca2 = PCA(n_components=2, random_state=RANDOM_STATE)
X_pca2 = pca2.fit_transform(X_scaled)

var_2d = pca2.explained_variance_ratio_
print(f"\n[5] PCA with 2 components:")
print(f"    Component 1: {var_2d[0]:.2%} variance")
print(f"    Component 2: {var_2d[1]:.2%} variance")
print(f"    Total: {sum(var_2d):.2%} variance explained")

# Apply PCA for 3D visualization
pca3 = PCA(n_components=3, random_state=RANDOM_STATE)
X_pca3 = pca3.fit_transform(X_scaled)

var_3d = pca3.explained_variance_ratio_
print(f"\n[6] PCA with 3 components:")
print(f"    Component 1: {var_3d[0]:.2%} variance")
print(f"    Component 2: {var_3d[1]:.2%} variance")
print(f"    Component 3: {var_3d[2]:.2%} variance")
print(f"    Total: {sum(var_3d):.2%} variance explained")

# ============================================================================
# SECTION 3: OPTIMAL K SELECTION USING ELBOW + SILHOUETTE
# ============================================================================

print(f"\n[7] Finding optimal number of clusters...")

K_range = range(2, 7)  # Test K from 2 to 6
results_2d = []
results_3d = []

for k in K_range:
    # Test on 2D PCA
    kmeans_2d = KMeans(n_clusters=k, init='k-means++', n_init=10, 
                       random_state=RANDOM_STATE)
    labels_2d = kmeans_2d.fit_predict(X_pca2)
    
    # Use sampling for silhouette on large dataset
    sample_size = min(10000, len(X_pca2))
    sil_2d = silhouette_score(X_pca2, labels_2d, sample_size=sample_size)
    inertia_2d = kmeans_2d.inertia_
    
    results_2d.append({
        'K': k,
        'Silhouette': sil_2d,
        'Inertia': inertia_2d
    })
    
    # Test on 3D PCA
    kmeans_3d = KMeans(n_clusters=k, init='k-means++', n_init=10,
                       random_state=RANDOM_STATE)
    labels_3d = kmeans_3d.fit_predict(X_pca3)
    
    sil_3d = silhouette_score(X_pca3, labels_3d, sample_size=sample_size)
    inertia_3d = kmeans_3d.inertia_
    
    results_3d.append({
        'K': k,
        'Silhouette': sil_3d,
        'Inertia': inertia_3d
    })

df_results_2d = pd.DataFrame(results_2d)
df_results_3d = pd.DataFrame(results_3d)

print("\n    Results for 2D PCA:")
print(df_results_2d.to_string(index=False))

print("\n    Results for 3D PCA:")
print(df_results_3d.to_string(index=False))

# Select optimal K (highest silhouette score)
optimal_k_2d = df_results_2d.loc[df_results_2d['Silhouette'].idxmax(), 'K']
optimal_k_3d = df_results_3d.loc[df_results_3d['Silhouette'].idxmax(), 'K']

print(f"\n[8] Optimal K selected:")
print(f"    2D PCA: K = {optimal_k_2d}")
print(f"    3D PCA: K = {optimal_k_3d}")

# ============================================================================
# SECTION 4: ELBOW METHOD VISUALIZATION
# ============================================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot for 2D
axes[0].plot(df_results_2d['K'], df_results_2d['Inertia'], 
             marker='o', linewidth=2, markersize=8, color='steelblue')
axes[0].set_xlabel('Number of Clusters (K)', fontsize=11)
axes[0].set_ylabel('Inertia (Within-Cluster Sum of Squares)', fontsize=11)
axes[0].set_title('Elbow Method - 2D PCA (Diabetes)', fontsize=12, fontweight='bold')
axes[0].grid(True, alpha=0.3)
axes[0].axvline(x=optimal_k_2d, color='red', linestyle='--', alpha=0.7, label=f'Optimal K={optimal_k_2d}')
axes[0].legend()

# Plot for 3D
axes[1].plot(df_results_3d['K'], df_results_3d['Inertia'],
             marker='o', linewidth=2, markersize=8, color='darkorange')
axes[1].set_xlabel('Number of Clusters (K)', fontsize=11)
axes[1].set_ylabel('Inertia (Within-Cluster Sum of Squares)', fontsize=11)
axes[1].set_title('Elbow Method - 3D PCA (Diabetes)', fontsize=12, fontweight='bold')
axes[1].grid(True, alpha=0.3)
axes[1].axvline(x=optimal_k_3d, color='red', linestyle='--', alpha=0.7, label=f'Optimal K={optimal_k_3d}')
axes[1].legend()

plt.tight_layout()
plt.savefig('diabetes_elbow_method.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\n[9] Elbow method plots saved as 'diabetes_elbow_method.png'")

# ============================================================================
# SECTION 5: FINAL K-MEANS++ WITH OPTIMAL K
# ============================================================================

# Train final models with optimal K
kmeans_final_2d = KMeans(n_clusters=int(optimal_k_2d), init='k-means++', 
                         n_init=10, random_state=RANDOM_STATE)
labels_final_2d = kmeans_final_2d.fit_predict(X_pca2)
centroids_2d = kmeans_final_2d.cluster_centers_

kmeans_final_3d = KMeans(n_clusters=int(optimal_k_3d), init='k-means++',
                         n_init=10, random_state=RANDOM_STATE)
labels_final_3d = kmeans_final_3d.fit_predict(X_pca3)
centroids_3d = kmeans_final_3d.cluster_centers_

print(f"\n[10] Final K-Means++ models trained")

# ============================================================================
# SECTION 6: CLUSTERING EVALUATION METRICS
# ============================================================================

def compute_metrics(X, labels, sample_size=10000):
    """Compute clustering evaluation metrics with sampling for large datasets"""
    if len(X) > sample_size:
        # Sample for silhouette score
        idx = np.random.choice(len(X), sample_size, replace=False)
        X_sample = X[idx]
        labels_sample = labels[idx]
        sil = silhouette_score(X_sample, labels_sample)
    else:
        sil = silhouette_score(X, labels)
    
    return {
        'Silhouette Score': sil,
        'Calinski-Harabasz Index': calinski_harabasz_score(X, labels),
        'Davies-Bouldin Index': davies_bouldin_score(X, labels)
    }

metrics_2d = compute_metrics(X_pca2, labels_final_2d)
metrics_3d = compute_metrics(X_pca3, labels_final_3d)

print(f"\n[11] Clustering Metrics:")
print(f"\n    2D PCA (K={int(optimal_k_2d)}):")
for metric, value in metrics_2d.items():
    print(f"      {metric}: {value:.4f}")

print(f"\n    3D PCA (K={int(optimal_k_3d)}):")
for metric, value in metrics_3d.items():
    print(f"      {metric}: {value:.4f}")

# Cluster size distribution
cluster_counts_2d = pd.Series(labels_final_2d).value_counts().sort_index()
cluster_counts_3d = pd.Series(labels_final_3d).value_counts().sort_index()

print(f"\n[12] Cluster Size Distribution:")
print(f"\n    2D PCA:")
for cluster_id, count in cluster_counts_2d.items():
    print(f"      Cluster {cluster_id+1}: {count} samples ({count/len(labels_final_2d)*100:.1f}%)")

print(f"\n    3D PCA:")
for cluster_id, count in cluster_counts_3d.items():
    print(f"      Cluster {cluster_id+1}: {count} samples ({count/len(labels_final_3d)*100:.1f}%)")

# ============================================================================
# SECTION 7: 2D VISUALIZATION
# ============================================================================

# Sample data for clearer visualization
n_show = min(5000, len(X_pca2))
idx_sample = np.random.choice(len(X_pca2), n_show, replace=False)

fig, ax = plt.subplots(figsize=(10, 7))

# Create discrete colormap
n_clusters_2d = int(optimal_k_2d)
colors = plt.cm.tab10(np.linspace(0, 1, n_clusters_2d))

# Plot each cluster
for i in range(n_clusters_2d):
    mask = labels_final_2d[idx_sample] == i
    ax.scatter(X_pca2[idx_sample][mask, 0], X_pca2[idx_sample][mask, 1],
               c=[colors[i]], label=f'Cluster {i+1}',
               s=15, alpha=0.6, edgecolors='k', linewidth=0.2)

# Plot centroids
ax.scatter(centroids_2d[:, 0], centroids_2d[:, 1],
           c='red', marker='X', s=300, edgecolors='black',
           linewidths=2, label='Centroids', zorder=10)

ax.set_xlabel('Principal Component 1', fontsize=12)
ax.set_ylabel('Principal Component 2', fontsize=12)
ax.set_title(f'Diabetes - K-Means++ Clustering (2D PCA, K={n_clusters_2d})',
             fontsize=14, fontweight='bold')
ax.legend(loc='best', framealpha=0.9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('diabetes_2d_clusters.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\n[13] 2D visualization saved as 'diabetes_2d_clusters.png'")

# ============================================================================
# SECTION 8: 3D VISUALIZATION
# ============================================================================

from mpl_toolkits.mplot3d import Axes3D

# Sample data for clearer 3D visualization
idx_sample_3d = np.random.choice(len(X_pca3), n_show, replace=False)

fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')

# Create discrete colormap
n_clusters_3d = int(optimal_k_3d)
colors_3d = plt.cm.tab10(np.linspace(0, 1, n_clusters_3d))

# Plot each cluster
for i in range(n_clusters_3d):
    mask = labels_final_3d[idx_sample_3d] == i
    ax.scatter(X_pca3[idx_sample_3d][mask, 0], 
               X_pca3[idx_sample_3d][mask, 1], 
               X_pca3[idx_sample_3d][mask, 2],
               c=[colors_3d[i]], label=f'Cluster {i+1}',
               s=10, alpha=0.6, edgecolors='k', linewidth=0.1)

# Plot centroids
ax.scatter(centroids_3d[:, 0], centroids_3d[:, 1], centroids_3d[:, 2],
           c='red', marker='X', s=400, edgecolors='black',
           linewidths=2.5, label='Centroids', zorder=10)

ax.set_xlabel('Principal Component 1', fontsize=11, labelpad=10)
ax.set_ylabel('Principal Component 2', fontsize=11, labelpad=10)
ax.set_zlabel('Principal Component 3', fontsize=11, labelpad=10)
ax.set_title(f'Diabetes - K-Means++ Clustering (3D PCA, K={n_clusters_3d})',
             fontsize=13, fontweight='bold', pad=20)

# Adjust viewing angle
ax.view_init(elev=20, azim=45)

# Set equal aspect ratio
max_range = np.abs(X_pca3[idx_sample_3d]).max() * 1.1
ax.set_xlim([-max_range, max_range])
ax.set_ylim([-max_range, max_range])
ax.set_zlim([-max_range, max_range])

ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), framealpha=0.9)

plt.tight_layout()
plt.savefig('diabetes_3d_clusters.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\n[14] 3D visualization saved as 'diabetes_3d_clusters.png'")

# ============================================================================
# SECTION 9: SUMMARY
# ============================================================================

print("\n" + "="*60)
print("ANALYSIS COMPLETE!")
print("="*60)
print(f"\nDataset: Diabetes ({X_raw.shape[0]} samples, {X_raw.shape[1]} features)")
print(f"\nOptimal Clusters:")
print(f"  - 2D PCA: K = {int(optimal_k_2d)}")
print(f"  - 3D PCA: K = {int(optimal_k_3d)}")
print(f"\nBest Silhouette Scores:")
print(f"  - 2D: {metrics_2d['Silhouette Score']:.4f}")
print(f"  - 3D: {metrics_3d['Silhouette Score']:.4f}")
print(f"\nOutputs saved:")
print(f"  1. diabetes_elbow_method.png")
print(f"  2. diabetes_2d_clusters.png")
print(f"  3. diabetes_3d_clusters.png")
print("="*60)
