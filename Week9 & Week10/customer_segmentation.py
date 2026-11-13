# ============================================================
# CUSTOMER SEGMENTATION USING MACHINE LEARNING
# Algorithms: K-Means, Hierarchical Clustering, DBSCAN
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage


# ============================================================
# 1. CREATE DATASET
# ============================================================

np.random.seed(42)

data = {
    "CustomerID": range(1, 201),
    "Age": np.random.randint(18, 65, 200),
    "Annual_Income": np.random.randint(20000, 120000, 200),
    "Spending_Score": np.random.randint(1, 101, 200)
}

df = pd.DataFrame(data)


# ============================================================
# 2. UNDERSTAND THE DATA
# ============================================================

print("=" * 60)
print("1. DATASET OVERVIEW")
print("=" * 60)

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset Information:")
df.info()

print("\nMissing Values:")
print(df.isnull().sum())

print("\nStatistical Summary:")
print(df.describe())

print("\nColumns:")
print(df.columns)

print("\nDataset Shape:")
print(df.shape)


# ============================================================
# 3. SELECT FEATURES
# ============================================================

X = df[
    ["Age", "Annual_Income", "Spending_Score"]
]

print("\n" + "=" * 60)
print("2. SELECTED FEATURES")
print("=" * 60)

print(X.head())


# ============================================================
# 4. DATA PREPROCESSING
# ============================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("\n" + "=" * 60)
print("3. SCALED DATA")
print("=" * 60)

print(X_scaled[:5])


# ============================================================
# 5. K-MEANS USING SCIKIT-LEARN PIPELINE
# ============================================================

print("\n" + "=" * 60)
print("4. K-MEANS CLUSTERING")
print("=" * 60)

kmeans_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("kmeans", KMeans(
        n_clusters=3,
        random_state=42,
        n_init=10
    ))
])

kmeans_labels = kmeans_pipeline.fit_predict(X)

df["KMeans_Cluster"] = kmeans_labels

print("\nK-Means Results:")
print(
    df[
        [
            "CustomerID",
            "Age",
            "Annual_Income",
            "Spending_Score",
            "KMeans_Cluster"
        ]
    ].head(10)
)


# ============================================================
# 6. ELBOW METHOD
# ============================================================

print("\n" + "=" * 60)
print("5. ELBOW METHOD")
print("=" * 60)

inertia = []

for k in range(2, 11):

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    model.fit(X_scaled)

    inertia.append(model.inertia_)

plt.figure(figsize=(8, 5))

plt.plot(
    range(2, 11),
    inertia,
    marker="o"
)

plt.title("Elbow Method for Optimal Number of Clusters")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")

plt.grid(True)
plt.show()


# ============================================================
# 7. SILHOUETTE SCORE FOR K-MEANS
# ============================================================

print("\n" + "=" * 60)
print("6. K-MEANS SILHOUETTE SCORES")
print("=" * 60)

for k in range(2, 7):

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(X_scaled)

    score = silhouette_score(
        X_scaled,
        labels
    )

    print(
        f"K = {k}, Silhouette Score = {score:.3f}"
    )


kmeans_score = silhouette_score(
    X_scaled,
    kmeans_labels
)

print(
    f"\nFinal K-Means Silhouette Score: "
    f"{kmeans_score:.3f}"
)


# ============================================================
# 8. K-MEANS VISUALIZATION
# ============================================================

plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=df,
    x="Annual_Income",
    y="Spending_Score",
    hue="KMeans_Cluster",
    palette="viridis",
    s=80
)

plt.title("K-Means Customer Segmentation")
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.legend(title="Cluster")

plt.show()


# ============================================================
# 9. HIERARCHICAL CLUSTERING - DENDROGRAM
# ============================================================

print("\n" + "=" * 60)
print("7. HIERARCHICAL CLUSTERING")
print("=" * 60)

linkage_matrix = linkage(
    X_scaled,
    method="ward"
)

plt.figure(figsize=(12, 6))

dendrogram(
    linkage_matrix,
    truncate_mode="lastp",
    p=20
)

plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Customer Groups")
plt.ylabel("Distance")

plt.show()


# ============================================================
# 10. HIERARCHICAL CLUSTERING MODEL
# ============================================================

hierarchical_model = AgglomerativeClustering(
    n_clusters=3,
    linkage="ward"
)

hierarchical_labels = hierarchical_model.fit_predict(
    X_scaled
)

df["Hierarchical_Cluster"] = hierarchical_labels

print("\nHierarchical Clustering Results:")

print(
    df[
        [
            "CustomerID",
            "Age",
            "Annual_Income",
            "Spending_Score",
            "Hierarchical_Cluster"
        ]
    ].head(10)
)


# ============================================================
# 11. HIERARCHICAL SILHOUETTE SCORE
# ============================================================

hierarchical_score = silhouette_score(
    X_scaled,
    hierarchical_labels
)

print(
    f"\nHierarchical Silhouette Score: "
    f"{hierarchical_score:.3f}"
)


# ============================================================
# 12. HIERARCHICAL VISUALIZATION
# ============================================================

plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=df,
    x="Annual_Income",
    y="Spending_Score",
    hue="Hierarchical_Cluster",
    palette="Set2",
    s=80
)

plt.title("Hierarchical Customer Segmentation")
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.legend(title="Cluster")

plt.show()


# ============================================================
# 13. DBSCAN CLUSTERING
# ============================================================

print("\n" + "=" * 60)
print("8. DBSCAN CLUSTERING")
print("=" * 60)

dbscan_model = DBSCAN(
    eps=0.8,
    min_samples=5
)

dbscan_labels = dbscan_model.fit_predict(
    X_scaled
)

df["DBSCAN_Cluster"] = dbscan_labels

print("\nDBSCAN Results:")

print(
    df[
        [
            "CustomerID",
            "Age",
            "Annual_Income",
            "Spending_Score",
            "DBSCAN_Cluster"
        ]
    ].head(10)
)


# ============================================================
# 14. DBSCAN CLUSTER INFORMATION
# ============================================================

unique_clusters = set(dbscan_labels)

print("\nDBSCAN Cluster Labels:")
print(unique_clusters)

n_clusters_dbscan = len(
    set(dbscan_labels)
    - {-1}
)

n_noise = list(
    dbscan_labels
).count(-1)

print(
    f"\nNumber of DBSCAN clusters: "
    f"{n_clusters_dbscan}"
)

print(
    f"Number of noise points: "
    f"{n_noise}"
)


# ============================================================
# 15. DBSCAN SILHOUETTE SCORE
# ============================================================

# Silhouette score requires at least 2 clusters.
# Noise points (-1) are removed for evaluation.

dbscan_mask = dbscan_labels != -1

if (
    n_clusters_dbscan >= 2
    and dbscan_mask.sum() > n_clusters_dbscan
):

    dbscan_score = silhouette_score(
        X_scaled[dbscan_mask],
        dbscan_labels[dbscan_mask]
    )

    print(
        f"\nDBSCAN Silhouette Score: "
        f"{dbscan_score:.3f}"
    )

else:

    dbscan_score = None

    print(
        "\nDBSCAN Silhouette Score cannot "
        "be calculated because fewer than "
        "2 valid clusters were formed."
    )


# ============================================================
# 16. DBSCAN VISUALIZATION
# ============================================================

plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=df,
    x="Annual_Income",
    y="Spending_Score",
    hue="DBSCAN_Cluster",
    palette="tab10",
    s=80
)

plt.title("DBSCAN Customer Segmentation")
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.legend(title="Cluster")

plt.show()


# ============================================================
# 17. CLUSTER PROFILES - K-MEANS
# ============================================================

print("\n" + "=" * 60)
print("9. K-MEANS CLUSTER PROFILES")
print("=" * 60)

kmeans_profile = df.groupby(
    "KMeans_Cluster"
)[
    ["Age", "Annual_Income", "Spending_Score"]
].mean()

print(kmeans_profile)


# ============================================================
# 18. MODEL COMPARISON
# ============================================================

print("\n" + "=" * 60)
print("10. MODEL COMPARISON")
print("=" * 60)

print(
    f"K-Means Silhouette Score: "
    f"{kmeans_score:.3f}"
)

print(
    f"Hierarchical Silhouette Score: "
    f"{hierarchical_score:.3f}"
)

if dbscan_score is not None:

    print(
        f"DBSCAN Silhouette Score: "
        f"{dbscan_score:.3f}"
    )

else:

    print(
        "DBSCAN Silhouette Score: Not Available"
    )


# ============================================================
# 19. FINAL INTERPRETATION
# ============================================================

print("\n" + "=" * 60)
print("11. FINAL INTERPRETATION")
print("=" * 60)

print("""
Customer segmentation divides customers into groups
based on similar characteristics.

K-Means:
- Divides customers into a predefined number of clusters.
- Works well when clusters are relatively compact.

Hierarchical Clustering:
- Builds a hierarchy of customer groups.
- The dendrogram helps visualize how customers/groups merge.

DBSCAN:
- Groups points based on density.
- Can identify noise or outlier customers.
- Does not require the number of clusters to be specified.

StandardScaler:
- Standardizes the features before clustering.
- Prevents Annual Income from dominating because of its
  larger numerical scale.

Pipeline:
- Combines preprocessing and model building into one workflow.
- Helps make the machine-learning process cleaner and reproducible.
""")

print("\nPROJECT COMPLETED SUCCESSFULLY!")
