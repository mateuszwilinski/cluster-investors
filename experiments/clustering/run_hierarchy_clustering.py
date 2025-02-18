import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances, silhouette_score
from scipy.cluster.hierarchy import linkage, fcluster
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cdist
from scipy.spatial.distance import pdist
from sklearn.preprocessing import MinMaxScaler
from skstab.datasets import load_dataset
from sklearn.metrics import classification_report
from collections import defaultdict



z=2   ### Data index (z=0,1 or 2) 
best_method='ward'  ## The optimal linkage for hierarchical clustering
optimal_k=7         ## Optimal number of clusters

#### Defining agent class

def map_agent_type(x):
    return (
        "market_maker(1)" if x == 1 else
        "market_maker(2)" if x == 2 else
        "market_maker(3)" if x == 3 else
        "market_taker(1)" if x == 4 else
        "market_taker(2)" if x == 5 else
        "market_taker(3)" if x == 6 else
        "fundamentalist(1)" if x == 7 else
        "fundamentalist(2)" if x == 8 else
        "fundamentalist(3)" if x == 9 else
        "fundamentalist(4)" if x == 10 else
        "chartist(1)" if x == 11 else
        "chartist(2)" if x == 12 else
        "chartist(3)" if x == 13 else
        "chartist(4)" if x == 14 else
        "noise_trader(1)" if x == 15 else
        "unknown"
    )


os.chdir("/Data/")

######### READING THE DATA AND STORING THE GROUND TRUTH 

# Initialize empty lists to store the data
data_list = []
truth_list = []

# Loop through i = 1, 2, 3
for i in range(1, 4):
     # Read the files c_train_i, x_test_i, x_validation_i
    x_train = pd.read_csv(f"x_train_{i}.csv", sep=" ", header=None)
    x_test = pd.read_csv(f"x_test_{i}.csv", sep=" ", header=None)
    x_validation = pd.read_csv(f"x_validation_{i}.csv", sep=" ", header=None)
        # Calculate rows per simulation
    y_train = pd.read_csv(f"y_train_{i}.csv", sep=" ", header=None)
    y_test = pd.read_csv(f"y_test_{i}.csv", sep=" ", header=None)
    y_validation = pd.read_csv(f"y_validation_{i}.csv", sep=" ", header=None)        
    # Combine train and validation data
    combined_train = pd.concat([x_train, x_validation], axis=0)
    combined_truth = pd.concat([y_train, y_validation], axis=0)

    # Store the merged training data and test data in the list
    data_list.append({"train_data": combined_train, "test_data": x_test})
    truth_list.append({"train_data": combined_truth, "test_data": y_test})


# Access the training and testing data to be used
data_to_be_used = data_list[z]["train_data"]
data_to_be_applied = data_list[z]["test_data"]
# Access ground truth data
# Increment the truth values by 1 (convert to numpy arrays for direct computation)
gt = (truth_list[z]["train_data"].values + 1).astype(int)
gt_applied = (truth_list[z]["test_data"].values + 1).astype(int)

# Example print statements to verify data shapes
print("Training data shape:", data_to_be_used.shape)
print("Test data shape:", data_to_be_applied.shape)
print("Ground truth training data shape:", gt.shape)
print("Ground truth test data shape:", gt_applied.shape)


######### SCALING THE DATA you want to use
# Scale the training data
scaler = StandardScaler()
sav_scaled = scaler.fit_transform(data_to_be_used)  # Ensure the data is scaled
data_scale = pd.DataFrame(sav_scaled, columns=[f"feature_{j}" for j in range(data_to_be_used.shape[1])])  # Add proper column names
sav_scaled_test = scaler.fit_transform(data_to_be_applied)  # Ensure the data is scaled
data_test_scale = pd.DataFrame(sav_scaled_test, columns=[f"feature_{j}" for j in range(data_to_be_used.shape[1])])  # Add proper column names
linkage_matrix = linkage(data_scale, method=best_method)

### Hierarchical clusterin 
clusters_hierarchical = fcluster(linkage_matrix, t=optimal_k, criterion="maxclust")
# Create a DataFrame for train data clusters and agent types
agent_types_scale = [map_agent_type(x) for x in gt]

# Create a DataFrame for data_scale clusters and agent types
data_scale_composition = pd.DataFrame({
    "Cluster": clusters_hierarchical,
    "AgentType": agent_types_scale
})

# Group by cluster and summarize the composition
composition_summary_scale = data_scale_composition.groupby("Cluster")["AgentType"].value_counts()
composition_summary_scale_df = composition_summary_scale.reset_index(name="Count")

############### Assigning clusters to the test_data
# Step 1: Compute cluster centroids for the training data
cluster_labels = np.unique(clusters_hierarchical)
centroids = np.array([
    data_scale[clusters_hierarchical == label].mean(axis=0) for label in cluster_labels
])


# Step 3: Calculate distances of test data to cluster centroids
distances = cdist(data_test_scale, centroids)

# Step 4: Assign test data points to the nearest cluster
test_data_clusters = np.argmin(distances, axis=1) + 1  # Add 1 to match cluster labels (1-indexed)


# Map agent types from gt_applied
agent_types = [map_agent_type(x) for x in gt_applied]

# Create a DataFrame for test data clusters and agent types
test_cluster_composition = pd.DataFrame({
    "Cluster": test_data_clusters,
    "AgentType": agent_types
})

# Group by cluster and summarize the composition
composition_summary = test_cluster_composition.groupby("Cluster")["AgentType"].value_counts()
composition_summary_df = composition_summary.reset_index(name="Count")

# Pivot the data to get clusters as rows and agent types as columns
table = composition_summary_scale_df.pivot(index="Cluster", columns="AgentType", values="Count").fillna(0)

# Ensure the table has all 14 agent types (in case some are missing in certain clusters)
all_agent_types = composition_summary_scale_df["AgentType"].unique()
table = table.reindex(columns=all_agent_types, fill_value=0)

# Convert to integer values for clarity
table = table.astype(int)

# Reset index to include Cluster as a column
table.reset_index(inplace=True)
table
# Identify the maximum value in each column (excluding the 'Cluster' column)
max_values = table.iloc[:, 1:].max()
max_indices = table.iloc[:, 1:].idxmax()

# Compute fractions for each column
column_sums = table.iloc[:, 1:].sum()
fractions = table.iloc[:, 1:].div(column_sums, axis=1)

# Create a DataFrame to store the maximum row for each column and the corresponding fraction
max_info_df = pd.DataFrame({
    "Agent Type": table.columns[1:], 
    "Max Cluster": table.loc[max_indices.values, 'Cluster'].values, 
    "Max Value": max_values.values, 
    "Fraction": (max_values / column_sums).values
})

max_info_df
# Identify all clusters in the table
all_clusters = set(table['Cluster'].unique())

# Identify clusters missing in max_info_df
assigned_clusters = set(max_info_df["Max Cluster"])
missing_clusters = all_clusters - assigned_clusters

# List to store missing cluster entries
missing_rows = []

# Find the agent type with the highest fraction for each missing cluster
for cluster in missing_clusters:
    row_idx = table[table["Cluster"] == cluster].index[0]
    best_agent_type = table.iloc[row_idx].idxmax()
    best_value = table.loc[row_idx, best_agent_type]
    best_fraction = fractions.loc[row_idx, best_agent_type]
    
    # Store the missing cluster data
    missing_rows.append({
        "Agent Type": best_agent_type,
        "Max Cluster": cluster,
        "Max Value": best_value,
        "Fraction": best_fraction
    })

# Append missing rows using pd.concat()
if missing_rows:
    missing_df = pd.DataFrame(missing_rows)
    max_info_df = pd.concat([max_info_df, missing_df], ignore_index=True)

# Initialize a dictionary to store multiple clusters per agent type
ground_truth_mapping = defaultdict(list)

# Populate the mapping
for agent, cluster in zip(max_info_df["Agent Type"], max_info_df["Max Cluster"]):
    ground_truth_mapping[agent].append(cluster)

# Convert to a regular dictionary (optional)
ground_truth_mapping = dict(ground_truth_mapping)
ground_truth_mapping

# Step 1: Build a Graph of Connected Clusters
cluster_graph = defaultdict(set)

for agent, clusters in ground_truth_mapping.items():
    for cluster in clusters:
        cluster_graph[cluster].update(clusters)

# Step 2: Find Fully Connected Components (Groups of Merged Clusters)
visited = set()
cluster_groups = []

def dfs(cluster, group):
    """Perform DFS to find all connected clusters"""
    if cluster in visited:
        return
    visited.add(cluster)
    group.add(cluster)
    for neighbor in cluster_graph[cluster]:
        dfs(neighbor, group)

# Find all merged groups
for cluster in cluster_graph:
    if cluster not in visited:
        group = set()
        dfs(cluster, group)
        cluster_groups.append(group)

# Step 3: Create a Mapping for Cluster Merging
cluster_replacement = {}
for group in cluster_groups:
    merged_label = "_".join(map(str, sorted(group)))  # Convert {3,4,6} → "3_4_6"
    for cluster in group:
        cluster_replacement[str(cluster)] = merged_label

# Step 4: Replace Clusters in `composition_summary_df`
composition_summary_df["Merged Cluster"] = composition_summary_df["Cluster"].astype(str).map(
    lambda x: cluster_replacement.get(x, x)  # Replace with merged label or keep original
)

# Aggregate Counts for Merged Clusters
merged_composition_df = (
    composition_summary_df.groupby(["Merged Cluster", "AgentType"])
    .agg({"Count": "sum"})
    .reset_index()
)

# Step 5: Create True and Predicted Labels for Classification Report
true_labels = []
pred_labels = []
weights = []

for agent_type, true_clusters in ground_truth_mapping.items():  # true_clusters is now a list
    agent_clusters = merged_composition_df[merged_composition_df["AgentType"] == agent_type]
    
    for _, row in agent_clusters.iterrows():
        pred_cluster = row["Merged Cluster"]
        count = row["Count"]  # Support count for this agent type in this cluster

        # **Get the proper merged cluster name from the mapping**
        true_label_str = cluster_replacement.get(str(true_clusters[0]), str(true_clusters[0]))  # Assign true cluster
        pred_label_str = cluster_replacement.get(str(pred_cluster), str(pred_cluster))  # Assign predicted cluster

        true_labels.extend([true_label_str] * count)  # Assign merged clusters as true labels
        pred_labels.extend([pred_label_str] * count)  # Assign predicted merged clusters
        weights.extend([count] * count)  # Weighting by actual occurrence

# Step 6: Compute Weighted Report
clustering_results = classification_report(
    true_labels, pred_labels, zero_division=0, output_dict=True
)

# Convert to DataFrame for better visualization
clustering_df = pd.DataFrame(clustering_results).transpose()
clustering_df





