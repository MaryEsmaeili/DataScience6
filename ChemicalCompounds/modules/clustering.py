import numpy as np
import umap
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler

class Clustering:
    def __init__(self, n_clusters=8, n_neighbors=15, min_dist=0.1, n_components=2):
        """
        Clustering class using UMAP for dimensionality reduction.

        Args:
            n_clusters (int): Number of clusters for KMeans and Agglomerative Clustering.
            n_neighbors (int): Number of neighbors for UMAP.
            min_dist (float): Minimum distance between points in UMAP space.
            n_components (int): Number of UMAP output dimensions.
        """
        self.n_clusters = n_clusters
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.n_components = n_components

    def scale_and_umap(self, features_df):
        """
        Scales the features and reduces dimensionality using UMAP.

        Args:
            features_df (pd.DataFrame): The feature dataframe.

        Returns:
            np.ndarray: The UMAP-transformed features.
        """
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features_df)

        umap_reducer = umap.UMAP(
            n_neighbors=self.n_neighbors,
            min_dist=self.min_dist,
            n_components=self.n_components,
            random_state=42
        )
        umap_features = umap_reducer.fit_transform(scaled_features)

        print(f"UMAP reduced features to {umap_features.shape[1]} dimensions.")
        return umap_features

    def perform_clustering(self, data):
        """
        Performs clustering using KMeans, and Agglomerative Clustering.

        Args:
            data (np.ndarray): The transformed feature matrix.

        Returns:
            dict: A dictionary with clustering method names as keys and labels as values.
        """
        kmeans = KMeans(n_clusters=self.n_clusters, init="k-means++", random_state=42).fit(data)
        hac = AgglomerativeClustering(n_clusters=self.n_clusters).fit(data)

        return {
            "KMeans": kmeans.labels_,
            "HAC": hac.labels_,
        }

    def visualize_clusters(self, data, labels):
        """
        Visualizes clustering results using UMAP projection.

        Args:
            data (np.ndarray): The transformed feature matrix.
            labels (dict): Clustering results with method names as keys and cluster labels as values.
        """
        umap_reducer = umap.UMAP(n_neighbors=15, min_dist=0.2, n_components=2, random_state=42)
        umap_data = umap_reducer.fit_transform(data)

        for method, label in labels.items():
            plt.figure(figsize=(8, 6))
            plt.scatter(umap_data[:, 0], umap_data[:, 1], c=label, cmap="Spectral", s=10)
            plt.title(f"Clustering with {method}")
            plt.xlabel("UMAP Dimension 1")
            plt.ylabel("UMAP Dimension 2")
            plt.show()
