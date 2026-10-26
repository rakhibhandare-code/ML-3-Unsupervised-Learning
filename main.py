# ML-3: Unsupervised Learning
# K-Means Clustering and PCA on the Wine Dataset

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


# --------------------------------------------------
# 1. Load the dataset
# --------------------------------------------------

print("=" * 55)
print("ML-3: UNSUPERVISED LEARNING")
print("K-Means Clustering and PCA")
print("=" * 55)

wine = load_wine()

# Use only numerical features
X = pd.DataFrame(wine.data, columns=wine.feature_names)

print("\n1. DATASET INFORMATION")
print("-" * 55)
print("Dataset: Wine Dataset")
print("Number of samples:", X.shape[0])
print("Number of numerical features:", X.shape[1])
print("\nFirst five rows:")
print(X.head())


# --------------------------------------------------
# 2. Check and prepare the data
# --------------------------------------------------

print("\n2. CHECKING DATA")
print("-" * 55)

print("Missing values:", X.isnull().sum().sum())

# Fill missing numerical values if any exist
X = X.fillna(X.mean())

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Standardization completed.")
print("Scaled data shape:", X_scaled.shape)


# --------------------------------------------------
# 3. Elbow Method and Silhouette Score
# --------------------------------------------------

print("\n3. FINDING THE NUMBER OF CLUSTERS")
print("-" * 55)

inertia_values = []
silhouette_values = []
k_values = range(2, 11)

for k in k_values:
    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(X_scaled)

    inertia_values.append(model.inertia_)
    silhouette_values.append(silhouette_score(X_scaled, labels))

    print(
        f"k = {k} | "
        f"Inertia = {model.inertia_:.2f} | "
        f"Silhouette Score = {silhouette_values[-1]:.4f}"
    )

# Plot the Elbow Method
plt.figure(figsize=(8, 5))
plt.plot(
    list(k_values),
    inertia_values,
    marker="o"
)
plt.title("Elbow Method")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.xticks(list(k_values))
plt.grid(True)
plt.tight_layout()
plt.savefig("elbow_method.png", dpi=300)
plt.show()

# Plot Silhouette Scores
plt.figure(figsize=(8, 5))
plt.plot(
    list(k_values),
    silhouette_values,
    marker="o"
)
plt.title("Silhouette Score for Different k Values")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Silhouette Score")
plt.xticks(list(k_values))
plt.grid(True)
plt.tight_layout()
plt.savefig("silhouette_scores.png", dpi=300)
plt.show()

# Select the k with the highest silhouette score
best_k = list(k_values)[int(np.argmax(silhouette_values))]

print("\nSelected number of clusters:", best_k)
print("Selection method: Highest silhouette score")


# --------------------------------------------------
# 4. Train the final K-Means model
# --------------------------------------------------

print("\n4. K-MEANS CLUSTERING")
print("-" * 55)

kmeans = KMeans(
    n_clusters=best_k,
    random_state=42,
    n_init=10
)

cluster_labels = kmeans.fit_predict(X_scaled)

X_result = X.copy()
X_result["Cluster"] = cluster_labels

print("K-Means clustering completed.")
print("\nNumber of samples in each cluster:")
print(X_result["Cluster"].value_counts().sort_index())

print(
    "\nFinal silhouette score:",
    round(silhouette_score(X_scaled, cluster_labels), 4)
)


# --------------------------------------------------
# 5. Reduce dimensions using PCA
# --------------------------------------------------

print("\n5. PRINCIPAL COMPONENT ANALYSIS (PCA)")
print("-" * 55)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

explained_variance = pca.explained_variance_ratio_

print("PCA completed.")
print("Original number of features:", X_scaled.shape[1])
print("Reduced number of features:", X_pca.shape[1])
print(
    f"PC1 explained variance: {explained_variance[0] * 100:.2f}%"
)
print(
    f"PC2 explained variance: {explained_variance[1] * 100:.2f}%"
)
print(
    f"Total variance explained by two components: "
    f"{sum(explained_variance) * 100:.2f}%"
)


# --------------------------------------------------
# 6. Visualize the clusters using PCA
# --------------------------------------------------

print("\n6. CREATING PCA CLUSTER VISUALIZATION")
print("-" * 55)

plt.figure(figsize=(9, 6))

scatter = plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=cluster_labels,
    cmap="viridis",
    s=60,
    edgecolors="black",
    alpha=0.8
)

plt.title("K-Means Clusters Visualized Using PCA")
plt.xlabel(
    f"Principal Component 1 ({explained_variance[0] * 100:.2f}% variance)"
)
plt.ylabel(
    f"Principal Component 2 ({explained_variance[1] * 100:.2f}% variance)"
)

plt.colorbar(scatter, label="Cluster")
plt.grid(True)
plt.tight_layout()
plt.savefig("pca_clusters.png", dpi=300)
plt.show()


# --------------------------------------------------
# 7. Interpret the clusters
# --------------------------------------------------

print("\n7. CLUSTER INTERPRETATION")
print("-" * 55)

# Calculate the mean of the original features for each cluster
cluster_means = X_result.groupby("Cluster").mean()

print("\nAverage feature values for each cluster:")
print(cluster_means.round(2).to_string())

# Find the most different features between clusters
feature_ranges = cluster_means.max(axis=0) - cluster_means.min(axis=0)

print("\nFeatures with the largest differences between cluster means:")
print(feature_ranges.sort_values(ascending=False).head(5).round(2))

print(
    "\nInterpretation: K-Means groups wine samples based on "
    "similarities across their numerical chemical features."
)
print(
    "The cluster mean values help identify which features "
    "differ most between the groups."
)
print(
    "PCA displays the high-dimensional data in two dimensions; "
    "some information is lost during dimensionality reduction."
)


# --------------------------------------------------
# 8. Save the results
# --------------------------------------------------

print("\n8. SAVING RESULTS")
print("-" * 55)

X_result.to_csv("wine_clusters.csv", index=False)

pca_results = pd.DataFrame({
    "PC1": X_pca[:, 0],
    "PC2": X_pca[:, 1],
    "Cluster": cluster_labels
})

pca_results.to_csv("pca_results.csv", index=False)

print("Saved: wine_clusters.csv")
print("Saved: pca_results.csv")
print("Saved: elbow_method.png")
print("Saved: silhouette_scores.png")
print("Saved: pca_clusters.png")

print("\n" + "=" * 55)
print("PROJECT EXECUTION COMPLETED SUCCESSFULLY!")
print("=" * 55)