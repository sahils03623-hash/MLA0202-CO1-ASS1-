
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, FactorAnalysis, FastICA
from sklearn.mixture import GaussianMixture

# Dataset given in the question
data = pd.DataFrame({
    "Alcohol": [14.23, 13.20, 13.16, 14.37, 13.24, 14.20, 14.39, 14.06, 14.83, 13.86],
    "MalicAcid": [1.71, 1.78, 2.36, 1.95, 2.59, 1.76, 1.87, 2.15, 1.64, 1.35],
    "Ash": [2.43, 2.14, 2.67, 2.50, 2.87, 2.45, 2.45, 2.61, 2.17, 2.27],
    "Alcalinity": [15.6, 11.2, 18.6, 16.8, 21.0, 15.2, 14.6, 17.6, 14.0, 16.0],
    "Magnesium": [127, 100, 101, 113, 118, 112, 96, 121, 97, 98],
    "Phenols": [2.80, 2.65, 2.80, 3.85, 2.80, 3.27, 2.50, 2.60, 2.80, 2.98]
})

print("Original Dataset:")
print(data)

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(data)

print("\nStandardized Data:")
print(pd.DataFrame(X_scaled, columns=data.columns).round(3))

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print("\nPCA Output:")
print(pd.DataFrame(X_pca, columns=["PC1", "PC2"]).round(3))

print("\nPCA Explained Variance Ratio:")
print(pca.explained_variance_ratio_)

# Factor Analysis
fa = FactorAnalysis(n_components=2, random_state=42)
X_fa = fa.fit_transform(X_scaled)

print("\nFactor Analysis Output:")
print(pd.DataFrame(X_fa, columns=["Factor1", "Factor2"]).round(3))

# ICA
ica = FastICA(n_components=2, random_state=42, max_iter=2000)
X_ica = ica.fit_transform(X_scaled)

print("\nICA Output:")
print(pd.DataFrame(X_ica, columns=["IC1", "IC2"]).round(3))

# Gaussian Mixture Model using EM Algorithm
gmm = GaussianMixture(
    n_components=2,
    random_state=42
)

clusters = gmm.fit_predict(X_pca)

print("\nGMM Cluster Labels:")
print(clusters)

# Visualize PCA with GMM clusters
plt.figure(figsize=(7, 5))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, s=80)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("GMM Clustering after PCA")
plt.grid(True)
plt.show()

# Visualize Factor Analysis
plt.figure(figsize=(7, 5))
plt.scatter(X_fa[:, 0], X_fa[:, 1], c=clusters, s=80)

plt.xlabel("Factor 1")
plt.ylabel("Factor 2")
plt.title("Factor Analysis with GMM Clusters")
plt.grid(True)
plt.show()

# Visualize ICA
plt.figure(figsize=(7, 5))
plt.scatter(X_ica[:, 0], X_ica[:, 1], c=clusters, s=80)

plt.xlabel("Independent Component 1")
plt.ylabel("Independent Component 2")
plt.title("ICA with GMM Clusters")
plt.grid(True)
plt.show()
