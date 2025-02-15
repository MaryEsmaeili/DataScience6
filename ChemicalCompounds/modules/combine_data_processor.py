import numpy as np
from modules.clustering_with_outliers import ClusteringAndOutlierDetection
from modules.evaluation import Evaluation
from modules.feature_processor import FeatureProcessor
from modules.visualizer import Visualizer


class CombinedDatasetProcessor:
    @staticmethod
    def process_combined_dataset(combined_df, butina_cutoff=0.5, n_neighbors=20, min_dist=0.01):
        """Apply Butina + HAC clustering on the combined dataset."""
        print(f"Combined dataset shape: {combined_df.shape}")

        # Prepare features and fingerprints
        fingerprints = combined_df['fingerprints'].tolist()
        numerical_features = FeatureProcessor.prepare_features(combined_df)

        # Perform Butina clustering
        butina_clusters = ClusteringAndOutlierDetection.butina_clustering(fingerprints, cutoff=butina_cutoff)
        butina_labels = np.zeros(len(combined_df), dtype=int) - 1
        for cluster_idx, cluster in enumerate(butina_clusters):
            for idx in cluster:
                butina_labels[idx] = cluster_idx
        combined_df['butina_cluster'] = butina_labels

        # Reduce dimensions with UMAP
        reduced_features = FeatureProcessor.apply_umap(numerical_features, n_neighbors=n_neighbors, min_dist=min_dist)

        # Apply HAC on UMAP-reduced features
        hac_labels = ClusteringAndOutlierDetection.apply_hac(reduced_features, n_clusters=None, distance_threshold=1.5)
        combined_df['hac_cluster'] = hac_labels

        # Visualize Butina clusters
        Visualizer.visualize_clusters(reduced_features, butina_labels, "Combined Dataset Clusters (Butina)")

        # Visualize HAC clusters
        Visualizer.visualize_clusters(reduced_features, hac_labels, "Combined Dataset Clusters (HAC)")

        # Validation Scores for Butina + HAC
        # Evaluate Clustering
        evaluation = Evaluation()
        evaluation.evaluate(umap_features, labels)

        return combined_df, butina_labels, hac_labels, reduced_features
