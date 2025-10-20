# Data-Decomposition-using-Clustering-Analysis
Data Decomposition using Clustering Analysis

This project explores multiple unsupervised learning algorithms to uncover hidden patterns in diverse datasets across health, physical fitness, and socioeconomic domains. Using K-Means, K-Means++, Agglomerative Hierarchical Clustering (AHC), and Gaussian Mixture Models (GMM), we compare the performance and interpretability of each method through multiple evaluation metrics.

## Key Highlights

Implemented clustering on Diabetes Prediction, Body Fat Prediction, and World Bank Development datasets from Kaggle.

Compared distance metrics (Euclidean vs Cosine) and dimensionality reduction techniques (PCA before/after clustering).

Evaluated model quality using:

1) Silhouette Score (cluster separation)

2) Davies–Bouldin Index (compactness)

3) Calinski–Harabasz Index (density)

4) Inertia (within-cluster variance)

5) Determined optimal K values through quantitative metrics and visual inspection (elbow and silhouette analysis).

6) Conducted extensive comparative experiments between algorithms for accuracy, stability, and computational efficiency.

## Findings

K-Means++ consistently outperforms standard K-Means in convergence speed and cluster quality.

Agglomerative Hierarchical Clustering (Ward linkage) performs better on smaller datasets with clear structure.

GMM offers interpretability through probabilistic clustering but is more sensitive to initialization and covariance type.

Across all datasets, K=2–3 often yields the most meaningful segmentation.

## Tech Stack

Python (Scikit-Learn, NumPy, Pandas, Matplotlib, Seaborn)

Jupyter Notebook for visualization and experimentation

PCA and distance metric customization for pre- and post-clustering analysis

## Datasets

[World Bank Dataset](https://www.kaggle.com/datasets/bhadramohit/world-bank-dataset)

[Diabetes Prediction Dataset](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset)

[Bodyfat Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/body-fat-prediction-dataset)

## Team Members
| Name | Email |
| ------------- | ------------- |
|Allen Lu Zhao Quan|ALLE0002@e.ntu.edu.sg|
|Gao Xinyue|GAOX0032@e.ntu.edu.sg|
|Hilda Tio|HTIO001@e.ntu.edu.sg|
|Tio Sher Min|STIO002@e.ntu.edu.sg|

## Contributors
| Component | Name |
| ------------- | ------------- |
|K-means|Allen, Sher Min, Xinyue|
|K++-means|Allen, Hilda, Xinyue|
|AHC|Allen, Hilda, Sher Min|
|GMM|Allen|
