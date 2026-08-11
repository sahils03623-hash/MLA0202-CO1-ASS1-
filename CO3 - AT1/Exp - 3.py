# Experiment 3: Dimensionality Reduction using PCA, Factor Analysis and ICA

import matplotlib.pyplot as plt

from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, FactorAnalysis, FastICA

# 1. Load Wine dataset
wine = load_wine()

X = wine.data
y = wine.target

print("Original data shape:", X.shape)

# 2. Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print("\nPCA Output Shape:", X_pca.shape)
print("PCA Explained Variance Ratio:")
print(pca.explained_variance_ratio_)
print("PCA Total Explained Variance:",
      round(pca.explained_variance_ratio_.sum(), 4))

# 4. Apply Factor Analysis
fa = FactorAnalysis(n_components=2, random_state=42)
X_fa = fa.fit_transform(X_scaled)

print("\nFactor Analysis Output Shape:", X_fa.shape)

# 5. Apply Independent Component Analysis
ica = FastICA(
    n_components=2,
    random_state=42,
    max_iter=1000
)

X_ica = ica.fit_transform(X_scaled)

print("ICA Output Shape:", X_ica.shape)

# 6. Visualize PCA
plt.figure(figsize=(7, 5))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, s=40)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA - Wine Dataset")
plt.grid(True)
plt.show()

# 7. Visualize Factor Analysis
plt.figure(figsize=(7, 5))
plt.scatter(X_fa[:, 0], X_fa[:, 1], c=y, s=40)
plt.xlabel("Factor 1")
plt.ylabel("Factor 2")
plt.title("Factor Analysis - Wine Dataset")
plt.grid(True)
plt.show()

# 8. Visualize ICA
plt.figure(figsize=(7, 5))
plt.scatter(X_ica[:, 0], X_ica[:, 1], c=y, s=40)
plt.xlabel("Independent Component 1")
plt.ylabel("Independent Component 2")
plt.title("ICA - Wine Dataset")
plt.grid(True)
plt.show()
