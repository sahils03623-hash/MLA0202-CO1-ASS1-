# Experiment 2: Clustering using EM Algorithm and Gaussian Mixture Model

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score

# 1. Load Digits dataset
digits = load_digits()

X = digits.data
y = digits.target

print("Dataset shape:", X.shape)

# 2. Preprocess the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Number of clusters
n_clusters = 10

# 3. Gaussian Mixture Model (EM Algorithm)
gmm = GaussianMixture(
    n_components=n_clusters,
    random_state=42
)

gmm_labels = gmm.fit_predict(X_scaled)

# 4. K-Means Clustering
kmeans = KMeans(
    n_clusters=n_clusters,
    random_state=42,
    n_init=10
)

kmeans_labels = kmeans.fit_predict(X_scaled)

# 5. Evaluate GMM
gmm_silhouette = silhouette_score(X_scaled, gmm_labels)
gmm_ari = adjusted_rand_score(y, gmm_labels)

# 6. Evaluate K-Means
kmeans_silhouette = silhouette_score(X_scaled, kmeans_labels)
kmeans_ari = adjusted_rand_score(y, kmeans_labels)

print("\nClustering Evaluation")
print("---------------------")

print("GMM Silhouette Score:", round(gmm_silhouette, 4))
print("GMM Adjusted Rand Index:", round(gmm_ari, 4))

print("\nK-Means Silhouette Score:", round(kmeans_silhouette, 4))
print("K-Means Adjusted Rand Index:", round(kmeans_ari, 4))

# 7. Apply PCA for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print("\nPCA Explained Variance Ratio:")
print(pca.explained_variance_ratio_)

print("Total Explained Variance:",
      round(pca.explained_variance_ratio_.sum(), 4))

# 8. Visualize GMM clusters
plt.figure(figsize=(8, 6))

plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=gmm_labels,
    s=15
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("Digits Clustering using GMM (EM Algorithm)")
plt.grid(True)
plt.show()

# 9. Visualize K-Means clusters
plt.figure(figsize=(8, 6))

plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=kmeans_labels,
    s=15
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("Digits Clustering using K-Means")
plt.grid(True)
plt.show()
