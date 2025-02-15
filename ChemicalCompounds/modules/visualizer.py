from matplotlib import pyplot as plt
import numpy as np
from modules.mix_cluster_identifier import MixedClusterIdentifier

class Visualizer:
    @staticmethod
    def visualize_clusters(data, labels, title):
        """Visualize clustering results in UMAP-reduced space."""
        plt.figure(figsize=(10, 8))
        plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', s=50, alpha=0.7)
        plt.colorbar(label='Cluster')
        plt.title(title)
        plt.xlabel("UMAP Dimension 1")
        plt.ylabel("UMAP Dimension 2")
        plt.show()

    @staticmethod
    def plot_outliers(features, outlier_labels, title):
        """Plot outliers detected by Isolation Forest and Elliptic Envelope."""
        plt.figure(figsize=(10, 8))
        unique_labels = set(outlier_labels)
        colors = {"Inlier": "blue", "Isolation Only": "orange", "Elliptic Only": "green", "Both": "red"}
        for label in unique_labels:
            mask = [l == label for l in outlier_labels]
            plt.scatter(features[mask, 0], features[mask, 1], 
                        c=colors[label], label=label, alpha=0.7, s=50)
        plt.legend(title="Outlier Type")
        plt.title(title)
        plt.xlabel("UMAP Dimension 1")
        plt.ylabel("UMAP Dimension 2")
        plt.show()

    @staticmethod
    def visualize_synthetic_natural_common(data, labels, sources, mixed_clusters, title):
        """Visualize synthetic, natural, and common compounds in UMAP-reduced space."""
        plt.figure(figsize=(12, 10))
        for label in np.unique(labels):
            mask = labels == label
            color = ('purple' if label in mixed_clusters else
                     ('orange' if all(s == 'Synthetic' for s in sources[mask]) else
                      'green'))
            plt.scatter(data[mask, 0], data[mask, 1], c=color, alpha=0.7, s=50, label=None)

        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=10, label='Synthetic'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Natural'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='purple', markersize=10, label='Common')
        ]
        plt.legend(handles=legend_elements, title="Compound Types", loc='upper right')
        plt.title(title)
        plt.xlabel("UMAP Dimension 1")
        plt.ylabel("UMAP Dimension 2")
        plt.show()

    @staticmethod
    def plot_tanimoto_histogram(df_synthetic, df_natural):
        """Plot histogram comparing average Tanimoto distances for synthetic and natural compounds."""
        plt.figure(figsize=(10, 6))
        plt.hist(df_synthetic['avg_tanimoto'], bins=30, alpha=0.7, label='Synthetic', color='orange')
        plt.hist(df_natural['avg_tanimoto'], bins=30, alpha=0.7, label='Natural', color='green')
        plt.xlabel("Average Tanimoto Distance")
        plt.ylabel("Frequency")
        plt.title("Comparison of Average Tanimoto Distances")
        plt.legend()
        plt.show()

    @staticmethod
    def find_and_visualize_mixed_clusters(combined_df, reduced_features, cluster_column, title="Mixed Clusters"):
        """Identify and visualize clusters containing both synthetic and natural compounds."""
        mixed_clusters = MixedClusterIdentifier.identify_mixed_clusters(combined_df, cluster_column)
        print(f"\nMixed clusters (containing both Synthetic and Natural): {mixed_clusters}")

        # Visualize mixed clusters
        Visualizer.visualize_synthetic_natural_common(
            reduced_features,
            combined_df[cluster_column].values,
            combined_df['source'].values,
            mixed_clusters,
            title
        )