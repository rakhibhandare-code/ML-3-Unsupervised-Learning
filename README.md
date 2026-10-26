# ML-3: Unsupervised Learning — K-Means Clustering and PCA

## Project Overview

This project applies unsupervised learning techniques to the Wine dataset using K-Means clustering and Principal Component Analysis (PCA).

The goal is to group wine samples based on their numerical features, evaluate the clustering results, and visualize the clusters in two dimensions.

## Dataset

- **Dataset:** Wine dataset from Scikit-learn
- **Number of samples:** 178
- **Number of numerical features:** 13
- **Missing values:** 0

The clustering model uses the numerical features without using the known wine class labels.

## Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

## Project Workflow

1. Load and explore the Wine dataset.
2. Check for missing values.
3. Standardize the numerical features using StandardScaler.
4. Apply the Elbow Method to evaluate different cluster counts.
5. Calculate silhouette scores for cluster counts from 2 to 10.
6. Select the cluster count with the highest silhouette score.
7. Train the final K-Means clustering model.
8. Apply PCA to reduce the 13 features to two principal components.
9. Visualize the clusters and interpret the cluster means.
10. Save the clustering results and visualizations.

## Results

- **Selected number of clusters:** 3
- **Final silhouette score:** 0.2849
- **PC1 explained variance:** 36.20%
- **PC2 explained variance:** 19.21%
- **Total variance explained by two PCA components:** 55.41%

### Cluster Distribution

| Cluster | Number of Samples |
|---|---:|
| 0 | 65 |
| 1 | 51 |
| 2 | 62 |

### Interpretation

K-Means divided the wine samples into three groups based on similarities in their chemical features.

The cluster means show differences between groups. In this run, proline, magnesium, and color intensity were among the features with the largest differences between cluster means.

The silhouette score indicates some clustering structure, although the clusters are not strongly separated.

PCA makes the results easier to visualize, but the two components preserve only 55.41% of the dataset's total variance.

## Generated Files

- `elbow_method.png` — Elbow Method graph
- `silhouette_scores.png` — Silhouette scores for different cluster counts
- `pca_clusters.png` — Two-dimensional PCA cluster visualization
- `wine_clusters.csv` — Original numerical features with cluster assignments
- `pca_results.csv` — PCA coordinates with cluster assignments

## How to Run

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### 2. Navigate to the project folder

```bash
cd ML-3-Unsupervised-Learning
```

### 3. Create a virtual environment

```bash
python -m venv .venv
```

### 4. Activate the virtual environment on Windows PowerShell

```powershell
.venv\Scripts\Activate.ps1
```

### 5. Install the dependencies

```bash
python -m pip install -r requirements.txt
```

### 6. Run the program

```bash
python main.py
```

## Conclusion

This project demonstrates the application of K-Means clustering and PCA to an unlabeled numerical dataset. The Elbow Method and silhouette scores help evaluate cluster counts, while PCA provides a two-dimensional visualization of the clustering results.