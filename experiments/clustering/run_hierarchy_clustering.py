import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import classification_report, confusion_matrix
from collections import defaultdict
from scipy.spatial.distance import cdist
import pylab as py

# Define the data directory
DATA_DIR = "Data"

# Define agent type mapping
agents = [
    "market_maker(1)", "market_maker(2)", "market_maker(3)",
    "market_taker(1)", "market_taker(2)", "market_taker(3)",
    "fundamentalist(1)", "fundamentalist(2)", "fundamentalist(3)", "fundamentalist(4)",
    "chartist(1)", "chartist(2)", "chartist(3)", "chartist(4)", "noise_trader(1)"
]

def load_datasets(data_dir):
    data_list = []
    truth_list = []
    for i in range(1, 4):
        x_train = pd.read_csv(os.path.join(data_dir, f"x_train_{i}.csv"), sep=" ", header=None)
        x_test = pd.read_csv(os.path.join(data_dir, f"x_test_{i}.csv"), sep=" ", header=None)
        x_val = pd.read_csv(os.path.join(data_dir, f"x_validation_{i}.csv"), sep=" ", header=None)
        y_train = pd.read_csv(os.path.join(data_dir, f"y_train_{i}.csv"), sep=" ", header=None)
        y_test = pd.read_csv(os.path.join(data_dir, f"y_test_{i}.csv"), sep=" ", header=None)
        y_val = pd.read_csv(os.path.join(data_dir, f"y_validation_{i}.csv"), sep=" ", header=None)

        combined_train = pd.concat([x_train, x_val], axis=0)
        combined_truth = pd.concat([y_train, y_val], axis=0)

        data_list.append({"train_data": combined_train, "test_data": x_test})
        truth_list.append({"train_data": combined_truth, "test_data": y_test})
    return data_list, truth_list

def plot_confusion_matrix(matrix, case, z, optimal_k):
    res_cm = -(matrix.T / matrix.sum(1)).T
    res_cm = res_cm - 2 * np.diag(np.diag(res_cm))

    py.rcParams["font.family"] = "serif"
    py.rcParams["mathtext.fontset"] = "cm"
    py.rcParams['pdf.fonttype'] = 42
    py.rcParams['ps.fonttype'] = 42

    py.figure(figsize=(5, 4))
    py.imshow(res_cm, cmap='RdBu', vmin=-1., vmax=1.)
    cbar = py.colorbar(fraction=0.035)
    cbar.ax.tick_params(labelsize=18)
    py.xticks([])
    py.yticks([])
    py.tick_params(direction='in', top=True, right=True, pad=7)
    py.tight_layout(pad=0.1)

    filename = f"conf_matrix_case{case}_z{z}_k{optimal_k}.png"
    py.savefig(filename, dpi=300, bbox_inches='tight')

def main():
    data_list, truth_list = load_datasets(DATA_DIR)
    best_method = 'ward'
    cases = [1, 2]

    for case in cases:
        for z in range(3):
            if case == 1:
                data_to_be_used = data_list[z]["train_data"]
                data_to_be_applied = data_list[z]["test_data"]
            else:
                data_to_be_used = data_list[z]["train_data"].iloc[:, :9]
                data_to_be_applied = data_list[z]["test_data"].iloc[:, :9]

            gt = (truth_list[z]["train_data"].values + 1).astype(int)
            gt_applied = (truth_list[z]["test_data"].values + 1).astype(int)

            scaler = StandardScaler()
            data_scale = pd.DataFrame(scaler.fit_transform(data_to_be_used))
            data_test_scale = pd.DataFrame(scaler.fit_transform(data_to_be_applied))
            linkage_matrix = linkage(data_scale, method=best_method)

            optimal_ks = [9, 15] if z == 0 else [9, 14]
            for optimal_k in optimal_ks:
                clusters_hierarchical = fcluster(linkage_matrix, t=optimal_k, criterion="maxclust")
                agent_types_scale = [agents[x - 1] for x in gt.flatten()]
                data_scale_composition = pd.DataFrame({
                    "Cluster": clusters_hierarchical,
                    "AgentType": agent_types_scale
                })

                composition_summary_scale_df = (
                    data_scale_composition.groupby("Cluster")["AgentType"]
                    .value_counts()
                    .reset_index(name="Count")
                )

                cluster_labels = np.unique(clusters_hierarchical)
                centroids = np.array([
                    data_scale[clusters_hierarchical == label].mean(axis=0)
                    for label in cluster_labels
                ])

                distances = cdist(data_test_scale, centroids)
                test_data_clusters = np.argmin(distances, axis=1) + 1

                agent_types = [agents[x - 1] for x in gt_applied.flatten()]
                test_cluster_composition = pd.DataFrame({
                    "Cluster": test_data_clusters,
                    "AgentType": agent_types
                })

                composition_summary_df = (
                    test_cluster_composition.groupby("Cluster")["AgentType"]
                    .value_counts()
                    .reset_index(name="Count")
                )

                table = composition_summary_scale_df.pivot(index="Cluster", columns="AgentType", values="Count").fillna(0)
                all_agent_types = composition_summary_scale_df["AgentType"].unique()
                table = table.reindex(columns=all_agent_types, fill_value=0).astype(int).reset_index()

                max_values = table.iloc[:, 1:].max()
                max_indices = table.iloc[:, 1:].idxmax()
                column_sums = table.iloc[:, 1:].sum()
                fractions = table.iloc[:, 1:].div(column_sums, axis=1)

                max_info_df = pd.DataFrame({
                    "Agent Type": table.columns[1:], 
                    "Max Cluster": table.loc[max_indices.values, 'Cluster'].values, 
                    "Max Value": max_values.values, 
                    "Fraction": (max_values / column_sums).values
                })

                all_clusters = set(table['Cluster'].unique())
                assigned_clusters = set(max_info_df["Max Cluster"])
                missing_clusters = all_clusters - assigned_clusters

                missing_rows = []
                for cluster in missing_clusters:
                    row_idx = table[table["Cluster"] == cluster].index[0]
                    best_agent_type = table.iloc[row_idx].idxmax()
                    best_value = table.loc[row_idx, best_agent_type]
                    best_fraction = fractions.loc[row_idx, best_agent_type]
                    missing_rows.append({
                        "Agent Type": best_agent_type,
                        "Max Cluster": cluster,
                        "Max Value": best_value,
                        "Fraction": best_fraction
                    })

                if missing_rows:
                    missing_df = pd.DataFrame(missing_rows)
                    max_info_df = pd.concat([max_info_df, missing_df], ignore_index=True)

                ground_truth_mapping = defaultdict(list)
                for agent, cluster in zip(max_info_df["Agent Type"], max_info_df["Max Cluster"]):
                    ground_truth_mapping[agent].append(cluster)

                cluster_graph = defaultdict(set)
                for agent, clusters in ground_truth_mapping.items():
                    for cluster in clusters:
                        cluster_graph[cluster].update(clusters)

                visited = set()
                cluster_groups = []
                def dfs(cluster, group):
                    if cluster in visited:
                        return
                    visited.add(cluster)
                    group.add(cluster)
                    for neighbor in cluster_graph[cluster]:
                        dfs(neighbor, group)

                for cluster in cluster_graph:
                    if cluster not in visited:
                        group = set()
                        dfs(cluster, group)
                        cluster_groups.append(group)

                cluster_replacement = {}
                for group in cluster_groups:
                    merged_label = "_".join(map(str, sorted(group)))
                    for cluster in group:
                        cluster_replacement[str(cluster)] = merged_label

                composition_summary_df["Merged Cluster"] = composition_summary_df["Cluster"].astype(str).map(
                    lambda x: cluster_replacement.get(x, x)
                )

                merged_composition_df = (
                    composition_summary_df.groupby(["Merged Cluster", "AgentType"])
                    .agg({"Count": "sum"})
                    .reset_index()
                )

                true_labels = []
                pred_labels = []

                for agent_type, true_clusters in ground_truth_mapping.items():
                    agent_clusters = merged_composition_df[merged_composition_df["AgentType"] == agent_type]
                    for _, row in agent_clusters.iterrows():
                        pred_cluster = row["Merged Cluster"]
                        count = row["Count"]
                        true_label_str = cluster_replacement.get(str(true_clusters[0]), str(true_clusters[0]))
                        pred_label_str = cluster_replacement.get(str(pred_cluster), str(pred_cluster))
                        true_labels.extend([true_label_str] * count)
                        pred_labels.extend([pred_label_str] * count)

                clustering_results = classification_report(
                    true_labels, pred_labels, zero_division=0, output_dict=True
                )

                labels = sorted(set(true_labels + pred_labels))
                clustering_df = pd.DataFrame(clustering_results).transpose()
                print("Report", clustering_df)

                conf_matrix = confusion_matrix(true_labels, pred_labels, labels=labels)
                plot_confusion_matrix(conf_matrix, case, z, optimal_k)

# Entry point
if __name__ == "__main__":
    main()
