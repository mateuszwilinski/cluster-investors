# ------------------- DEPENDENCIES ------------------- #
# Install the required packages if not already available
# !pip install scikit-learn-extra
# !pip install git+https://github.com/FlorentF9/skstab.git

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering
from sklearn_extra.cluster import KMedoids
from tslearn.clustering import KernelKMeans

from scipy.cluster.hierarchy import linkage, fcluster
from skstab import StadionEstimator

# ------------------- CONFIGURATION ------------------- #
DATA_DIR = "Data"
NSIM = 40
Z_INDEX = 0           # 0 = no noise case
K_COMBINE = 5         # Number of simulations to combine
# ------------------- CONFIGURATION FOR STADION SCORES ------------------- #
RUNS = 20             
K_VALUES = list(range(1, 17))
OMEGA = list(range(2, 6))

METHOD = "kmeans"  # Options: kmeans, kmedoids, hierarchical, spectral, kernel_kmeans

# ------------------- DATA LOADING ------------------- #
def load_csv_file(name, index):
    return pd.read_csv(os.path.join(DATA_DIR, f"{name}_{index}.csv"), sep=" ", header=None)

def load_simulation_data(z_index):
    i = z_index + 1
    x_train = load_csv_file("x_train", i)
    x_test = load_csv_file("x_test", i)
    x_validation = load_csv_file("x_validation", i)
    y_train = load_csv_file("y_train", i)
    y_test = load_csv_file("y_test", i)
    y_validation = load_csv_file("y_validation", i)

    sim_list, truth_list = [], []
    sim_rows_train = x_train.shape[0] // NSIM
    sim_rows_val = x_validation.shape[0] // NSIM

    for j in range(NSIM):
        x_part = pd.concat([
            x_train.iloc[j * sim_rows_train:(j + 1) * sim_rows_train],
            x_validation.iloc[j * sim_rows_val:(j + 1) * sim_rows_val]
        ])
        y_part = np.concatenate([
            y_train.iloc[j * sim_rows_train:(j + 1) * sim_rows_train].values.flatten(),
            y_validation.iloc[j * sim_rows_val:(j + 1) * sim_rows_val].values.flatten()
        ])
        sim_list.append(x_part.values)
        truth_list.append(y_part)

    return sim_list, truth_list

# ------------------- DATA COMBINING & SCALING ------------------- #
def combine_simulations(sim_list, truth_list, k):
    selected = np.random.choice(len(sim_list), size=k, replace=False)
    combined_x = np.vstack([sim_list[i] for i in selected])
    combined_y = np.concatenate([truth_list[i] for i in selected])
    return combined_x, combined_y

def scale_data(data):
    scaler = StandardScaler()
    return pd.DataFrame(scaler.fit_transform(data), columns=[f"feature_{j}" for j in range(data.shape[1])])

# ------------------- STABILITY ANALYSIS ------------------- #
def run_stability_analysis(method, data_scale, k_values, omega, runs):
    X = data_scale.values
    crossing = False
    # ---------------- Standard methods ---------------- #
    if method == "kmeans":
        algorithm = KMeans
        algo_kwargs = {"init": "k-means++", "n_init": 10, "random_state": 42}

    elif method == "kmedoids":
        algorithm = KMedoids
        algo_kwargs = {"metric": "euclidean", "random_state": 42}

    elif method == "spectral":
        class SpectralClusteringWrapper:
            def __init__(self, affinity="rbf", random_state=42, eigen_tol=1e-4):
                self.affinity = affinity
                self.random_state = random_state
                self.eigen_tol = eigen_tol
                self.labels_ = None
                self.model = None

            def fit(self, X, **kwargs):
                n_clusters = kwargs.get("n_clusters", 2)
                self.model = SpectralClustering(
                    n_clusters=n_clusters,
                    affinity=self.affinity,
                    random_state=self.random_state,
                    eigen_tol=self.eigen_tol
                )
                self.labels_ = self.model.fit_predict(X)
                return self

            def predict(self, X):
                return self.labels_

        algorithm = lambda **kwargs: SpectralClusteringWrapper()
        algo_kwargs = {}

    elif method == "kernel_kmeans":
        class KernelKMeansWrapper:
            def __init__(self, **kwargs):
                self.model = KernelKMeans(**kwargs)

            def fit(self, X):
                if X.ndim == 2:
                    X = X[:, :, np.newaxis]
                self.labels_ = self.model.fit_predict(X)
                return self

            def predict(self, X):
                return self.labels_

        algorithm = lambda **kwargs: KernelKMeansWrapper(kernel="rbf", random_state=42)
        algo_kwargs = {}

    elif method == "hierarchical":
        class HierarchicalClusteringWrapper:
            def __init__(self, linkage_method="ward", criterion="maxclust"):
                self.linkage_method = linkage_method
                self.criterion = criterion

            def fit(self, X, **kwargs):
                n_clusters = kwargs.get("n_clusters", 2)
                self.linkage_matrix = linkage(X, method=self.linkage_method)
                self.labels_ = fcluster(self.linkage_matrix, t=n_clusters, criterion=self.criterion)
                return self

            def predict(self, X):
                return self.labels_

        estimator = StadionEstimator(
            X=X,
            algorithm=lambda **kwargs: HierarchicalClusteringWrapper(),
            param_name="n_clusters",
            param_values=k_values,
            omega=omega,
            perturbation="uniform",
            runs=runs,
            algo_kwargs={},
            n_jobs=-1
        )

    else:
        raise ValueError(f"Unknown method: {method}")

    # ---------------- Construct estimator for all except hierarchical ---------------- #
    if method != "hierarchical":
        estimator = StadionEstimator(
            X=X,
            algorithm=algorithm,
            param_name="n_clusters",
            param_values=k_values,
            omega=omega,
            perturbation="uniform",
            perturbation_kwargs="auto",
            runs=runs,
            algo_kwargs=algo_kwargs,
            extended=True,
            n_jobs=-1
        )

    # ---------------- Evaluate ---------------- #
    try:
        score_mean = estimator.score(strategy="mean", crossing=crossing)
        return score_mean
    except Exception as e:
        warnings.warn(f"Stability estimation failed for method '{method}': {e}")
        return None




# ------------------- MAIN EXECUTION ------------------- #
def main():
    # Load and combine simulations
    sim_list, truth_list = load_simulation_data(Z_INDEX)
    combined_data, combined_truth = combine_simulations(sim_list, truth_list, K_COMBINE)
    data_scale = scale_data(combined_data)

    # Run stability analysis (only mean scores returned)
    score_mean = run_stability_analysis(
        method=METHOD,
        data_scale=data_scale,
        k_values=K_VALUES,
        omega=OMEGA,
        runs=RUNS
    )

    # Display results
    pretty_method = METHOD.replace("_", " ").title()
    print(f"{pretty_method}: Stability Scores (Mean): {score_mean}")

