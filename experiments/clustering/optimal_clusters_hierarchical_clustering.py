import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from kneed import KneeLocator

# -------------------- Configuration -------------------- #
DATA_DIR = "Data"
Z_INDEX = 0  # Use z=0 only (No Noise data is used to obtain optimal number of clusters)
BEST_METHOD = 'ward'

# -------------------- Load Data -------------------- #
def load_single_dataset(data_dir, z_index):
    i = z_index + 1
    x_train = pd.read_csv(os.path.join(data_dir, f"x_train_{i}.csv"), sep=" ", header=None)
    x_test = pd.read_csv(os.path.join(data_dir, f"x_test_{i}.csv"), sep=" ", header=None)
    x_val = pd.read_csv(os.path.join(data_dir, f"x_validation_{i}.csv"), sep=" ", header=None)

    combined_train = pd.concat([x_train, x_val], axis=0)
    return combined_train.reset_index(drop=True)

# -------------------- Normalize Data and obtain Linkage matrix for Hierarchical clustering -------------------- #
def scale_and_link(data):
    scaler = StandardScaler()
    data_scaled = pd.DataFrame(scaler.fit_transform(data))
    linkage_matrix = linkage(data_scaled, method=BEST_METHOD)
    return data_scaled, linkage_matrix

# -------------------- Evaluate Clustering -------------------- #
def evaluate_clustering(data_scaled, linkage_matrix):
    sil_scores = []
    wcss_scores = []
    cluster_range = range(2, 17)

    for k in cluster_range:
        clusters = fcluster(linkage_matrix, t=k, criterion="maxclust")
        sil = silhouette_score(data_scaled, clusters)
        sil_scores.append(sil)

        centroids = np.array([data_scaled[clusters == label].mean(axis=0) for label in np.unique(clusters)])
        wcss = np.sum(np.linalg.norm(data_scaled - centroids[clusters - 1], axis=1) ** 2)
        wcss_scores.append(wcss)

    return list(cluster_range), sil_scores, wcss_scores

# -------------------- Plot Results -------------------- #
def plot_elbow_and_silhouette(cluster_range, sil_scores, wcss_scores):
    knee_locator = KneeLocator(cluster_range, wcss_scores, curve="convex", direction="decreasing")
    optimal_k_wcss = knee_locator.knee

    sil_filtered = sil_scores[3:]
    cluster_filtered = cluster_range[3:]
    optimal_k_sil = cluster_filtered[np.argmax(sil_filtered)]

    # Plot WCSS (Elbow)
    plt.figure(figsize=(5, 3))
    plt.plot(cluster_range, wcss_scores, marker='o', label='WCSS')
    plt.axvline(x=optimal_k_wcss, color='r', linestyle='--', label=f'Optimal k: {optimal_k_wcss}')
    plt.title("Elbow Plot (WCSS)")
    plt.xlabel("Number of Clusters")
    plt.ylabel("WCSS")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("wcss.pdf", bbox_inches='tight')
    plt.show()

    # Plot Silhouette
    plt.figure(figsize=(5, 3))
    plt.plot(cluster_filtered, sil_filtered, marker='s', color='g', label='Silhouette Score')
    plt.axvline(x=optimal_k_sil, color='r', linestyle='--', label=f'Max Silhouette k: {optimal_k_sil}')
    plt.title("Silhouette Score vs. Number of Clusters")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Silhouette Score")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("sil.pdf", bbox_inches='tight')
    plt.show()

    return optimal_k_wcss, optimal_k_sil

# -------------------- Main -------------------- #
def main():
    data = load_single_dataset(DATA_DIR, Z_INDEX)
    data_scaled, linkage_matrix = scale_and_link(data)
    cluster_range, sil_scores, wcss_scores = evaluate_clustering(data_scaled.values, linkage_matrix)
    optimal_k_wcss, optimal_k_sil = plot_elbow_and_silhouette(cluster_range, sil_scores, wcss_scores)

    print(f"Optimal k (WCSS): {optimal_k_wcss}")
    print(f"Optimal k (Silhouette): {optimal_k_sil}")

if __name__ == "__main__":
    main()
