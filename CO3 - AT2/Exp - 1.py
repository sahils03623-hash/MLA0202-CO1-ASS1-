
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# Dataset given in the question
data = pd.DataFrame({
    "CustomerID": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Age": [19, 21, 20, 23, 31, 22, 35, 23, 64, 30],
    "AnnualIncome": [15, 15, 16, 16, 17, 17, 18, 18, 19, 19],
    "SpendingScore": [39, 81, 6, 77, 40, 76, 6, 94, 3, 72]
})

# Select features
X = data[["AnnualIncome", "SpendingScore"]]

# Preprocess the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Elbow Method
inertia = []

for k in range(2, 6):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

plt.plot(range(2, 6), inertia, marker="o")
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.grid(True)
plt.show()

# Silhouette Score
scores = []

for k in range(2, 6):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    scores.append(score)
    print("K =", k, "Silhouette Score =", round(score, 4))

# Find optimal number of clusters
optimal_k = range(2, 6)[scores.index(max(scores))]

print("\nOptimal Number of Clusters:", optimal_k)

# Apply K-Means with optimal K
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)

data["Cluster"] = clusters

print("\nCustomer Segmentation:")
print(data)

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print("\nPCA Explained Variance Ratio:")
print(pca.explained_variance_ratio_)

# Visualize clusters using PCA
plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=clusters,
    s=80
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("Customer Segmentation using K-Means and PCA")
plt.grid(True)
plt.show()
