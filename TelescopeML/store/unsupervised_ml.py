# Import necessary libraries
import pandas as pd
import numpy as np
import os

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
%matplotlib inline

# Set random seed for reproducibility
np.random.seed(seed=100)

# Define the reference data path
__reference_data_path__ = os.getenv("TelescopeML_reference_data")

# If the environment variable is not set, manually set the path
if not __reference_data_path__:
    __reference_data_path__ = "D:/Nasa/TelescopeML_project/reference_data"

# Load the training dataset
train_BD = pd.read_csv(
    os.path.join(
        __reference_data_path__,
        'training_datasets',
        'browndwarf_R100_v4_newWL_v3.csv.bz2'
    ),
    compression='bz2'
)
train_BD

# Output variable names
output_names = ['gravity', 'temperature', 'c_o_ratio', 'metallicity']

# Show the output table setup
print("Output variables:")
display(train_BD[output_names].head())

# Input feature names (wavelengths)
wavelength_names = [col for col in train_BD.columns if col not in output_names]
print("\nInput features (wavelengths):")
print(wavelength_names[:5])

# Training feature variables (input features)
X = train_BD.drop(columns=output_names)
X

# Target/Output feature variables
y = train_BD[output_names]
y

# Print the number of input features and output variables
print("\nNumber of input features (wavelengths):", X.shape[1])
print("Number of output variables:", y.shape[1])

# Convert data types to float32
X = X.astype(np.float32)
y = y.astype(np.float32)

# Split data: 60% training, 20% cross-validation, 20% test
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
X_train, X_cv, y_train, y_cv = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=42  # 0.25 x 0.8 = 0.2
)

# Verify the sizes
assert len(X_train) + len(X_cv) + len(X_test) == len(X), "Data splitting error"
print(f"\nTraining set size: {len(X_train)}")
print(f"Cross-validation set size: {len(X_cv)}")
print(f"Test set size: {len(X_test)}")

# Function to compute quartile labels using training data thresholds
def compute_quartile_labels(y_df, quartile_thresholds=None):
    y_quartiles = y_df.copy()
    quartile_labels = {}
    for col in y_df.columns:
        if quartile_thresholds and col in quartile_thresholds:
            quartiles = quartile_thresholds[col]
        else:
            # Compute quartiles on the training data
            quartiles = np.quantile(y_train[col], [0.25, 0.5, 0.75])
        # Assign labels based on quartiles
        y_quartiles[col + '_quartile'] = pd.cut(
            y_df[col],
            bins=[-np.inf, quartiles[0], quartiles[1], quartiles[2], np.inf],
            labels=[0, 1, 2, 3]
        )
        quartile_labels[col] = quartiles
    return y_quartiles, quartile_labels

# Compute quartile labels and thresholds for training data
y_train_quartiles, quartile_thresholds = compute_quartile_labels(y_train)

# Display quartile thresholds
print("\nQuartile thresholds for output variables (training data):")
for col in output_names:
    print(f"{col}: {quartile_thresholds[col]}")

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_cv_scaled = scaler.transform(X_cv)
X_test_scaled = scaler.transform(X_test)

# Fit PCA on the training data without specifying n_components
pca_full = PCA()
pca_full.fit(X_train_scaled)

# Calculate cumulative explained variance
cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)

# Find the number of components that explain at least 95% variance
n_components = np.argmax(cumulative_variance >= 0.95) + 1  # Add 1 because index starts at 0

print(f"\nNumber of PCA components to retain 95% variance: {n_components}")

# Apply PCA with the optimal number of components
pca = PCA(n_components=n_components)
X_train_pca = pca.fit_transform(X_train_scaled)
X_cv_pca = pca.transform(X_cv_scaled)
X_test_pca = pca.transform(X_test_scaled)

print(f"PCA applied, reduced data shape: {X_train_pca.shape}")

# Elbow method to find optimal k
inertias = []
K = range(1, 11)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_train_pca)
    inertias.append(kmeans.inertia_)

# Plot the elbow curve
plt.figure(figsize=(8, 6))
plt.plot(K, inertias, 'bo-', markersize=8)
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method For Optimal k')
plt.grid(True)
plt.show()

# Option for the user to change the number of centroid clusters
optimal_k = int(input("Enter the optimal number of clusters based on the elbow plot: "))
print(f"\nOptimal number of clusters selected: {optimal_k}")

# =============================================================================
# Implement Custom K-Means with Visualization
# =============================================================================

def plot_progress_kMeans(X, centroids, previous_centroids, labels, K, iter_num):
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.6)
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=100, linewidths=3)
    for j in range(K):
        plt.plot([previous_centroids[j, 0], centroids[j, 0]],
                 [previous_centroids[j, 1], centroids[j, 1]], 'k-', lw=2)
    plt.title(f'Iteration {iter_num + 1}')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.grid(True)
    plt.show()
    plt.pause(0.1)
    plt.close()

def run_kMeans(X, K, max_iters=10):
    # Initialize centroids randomly from the dataset
    np.random.seed(42)
    indices = np.random.choice(X.shape[0], K, replace=False)
    centroids = X[indices]
    previous_centroids = centroids.copy()
    
    for i in range(max_iters):
        # Assign clusters using Euclidean distance
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        
        # Plot progress
        plot_progress_kMeans(X, centroids, previous_centroids, labels, K, i)
        
        # Update centroids
        previous_centroids = centroids.copy()
        for k in range(K):
            points = X[labels == k]
            if len(points) > 0:
                centroids[k] = points.mean(axis=0)
            else:
                # Reinitialize centroid if no points are assigned
                centroids[k] = X[np.random.choice(X.shape[0])]
    return centroids, labels

# Option for the user to change the number of iterations
max_iters = int(input("Enter the number of iterations for K-Means Algorithm: "))

# Run K-Means with the optimal number of clusters
centroids_final, train_cluster_labels = run_kMeans(X_train_pca, optimal_k, max_iters=max_iters)

# Add cluster labels to the training dataset
train_BD_train = X_train.copy()
train_BD_train['Cluster'] = train_cluster_labels

# Merge quartile labels with cluster labels
y_train_quartiles_with_clusters = y_train_quartiles.copy()
y_train_quartiles_with_clusters['Cluster'] = train_cluster_labels

# For each output variable, determine the majority quartile label in each cluster
cluster_quartile_labels = {}
for col in output_names:
    col_quartile = col + '_quartile'
    cluster_mode = y_train_quartiles_with_clusters.groupby('Cluster')[col_quartile].agg(
        lambda x: x.value_counts().index[0]
    )
    cluster_quartile_labels[col] = cluster_mode

# Plotting the majority quartile labels for each cluster
for col in output_names:
    # Get the majority quartile labels for the output variable
    cluster_mode = cluster_quartile_labels[col]
    clusters = cluster_mode.index
    quartile_labels = cluster_mode.values.astype(int)
    
    plt.figure(figsize=(8, 6))
    plt.bar(clusters, quartile_labels, color='skyblue')
    plt.xlabel('Cluster')
    plt.ylabel('Majority Quartile Label')
    plt.title(f'Majority Quartile Labels per Cluster for {col.capitalize()}')
    plt.xticks(clusters)
    plt.yticks([0, 1, 2, 3])
    plt.ylim(-0.5, 3.5)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

# Function to calculate accuracy and display results
def evaluate_model(X_data_pca, y_data, data_label):
    # Assign clusters to data
    distances = np.linalg.norm(X_data_pca[:, np.newaxis] - centroids_final, axis=2)
    cluster_labels = np.argmin(distances, axis=1)
    
    # Assign predicted quartile labels based on cluster assignments
    y_pred_quartiles = pd.DataFrame(index=y_data.index)
    for col in output_names:
        col_quartile = col + '_quartile'
        # Map cluster labels to quartile labels
        cluster_label_map = cluster_quartile_labels[col].to_dict()
        cluster_labels_series = pd.Series(cluster_labels, index=y_data.index)
        y_pred_quartiles[col_quartile] = cluster_labels_series.map(cluster_label_map)
    
    # Compute true quartile labels for data using training thresholds
    y_true_quartiles, _ = compute_quartile_labels(y_data, quartile_thresholds=quartile_thresholds)
    
    # Calculate prediction accuracy
    accuracy_scores = {}
    for col in output_names:
        col_quartile = col + '_quartile'
        y_true = y_true_quartiles[col_quartile].astype(int)
        y_pred = y_pred_quartiles[col_quartile].astype(int)
        accuracy = accuracy_score(y_true, y_pred)
        accuracy_scores[col] = accuracy
        print(f"\nAccuracy for {col} on {data_label} Data: {accuracy * 100:.2f}%")
    
    # Overall accuracy
    overall_accuracy = np.mean(list(accuracy_scores.values()))
    print(f"\nOverall Accuracy on {data_label} Data: {overall_accuracy * 100:.2f}%")
    return overall_accuracy

# Evaluate model on training data (optional)
# evaluate_model(X_train_pca, y_train, 'Training')

# Evaluate model on cross-validation data
overall_accuracy_cv = evaluate_model(X_cv_pca, y_cv, 'Cross-Validation')

# Evaluate model on test data
overall_accuracy_test = evaluate_model(X_test_pca, y_test, 'Test')

# Compare accuracy before and after PCA (Placeholder values for before PCA)
# Assuming you have the accuracy before PCA stored in a variable `accuracy_before_pca`
accuracy_before_pca = 0.50  # Placeholder value

print(f"\nAccuracy without PCA: {accuracy_before_pca * 100:.2f}%")
print(f"Accuracy with PCA on Test Data: {overall_accuracy_test * 100:.2f}%")

# Plotting accuracy comparison
plt.figure(figsize=(8, 6))
methods = ['Without PCA', 'With PCA']
accuracies = [accuracy_before_pca * 100, overall_accuracy_test * 100]
plt.bar(methods, accuracies, color=['red', 'green'])
plt.ylabel('Accuracy (%)')
plt.title('Accuracy Comparison Before and After PCA')
plt.ylim(0, 100)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
