from bleach import Cleaner
import pandas as pd

from modules.clustering_with_outliers import ClusteringAndOutlierDetection
from modules.descriptor_fingerprint import DescriptorFingerprintProcessor
from modules.evaluation import Evaluation
from modules.feature_processor import FeatureProcessor
from modules.visualizer import Visualizer


class MolecularPipeline:
    def __init__(self, data_path, dataset_name, source_type, butina_cutoff=0.5, n_neighbors=20, min_dist=0.01):
        self.data_path = data_path
        self.dataset_name = dataset_name
        self.source_type = source_type
        self.butina_cutoff = butina_cutoff
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist

    def process_and_cluster(self):
        """Full pipeline to clean, process, and cluster data."""
        # Load and clean data
        df = pd.read_csv(self.data_path, sep="\t", names=["name", "molecular_formula", "canonical_smiles"], header=None)
        print(f"{self.dataset_name} initial data shape: {df.shape}")  # Debugging checkpoint
        
        df_cleaned = Cleaner(df)  # Using the clean function to validate and clean the dataset
        print(f"{self.dataset_name} cleaned data shape: {df_cleaned.shape}")  # Debugging checkpoint
        df_cleaned['source'] = self.source_type

        # Compute descriptors
        descriptors = df_cleaned['canonical_smiles'].apply(DescriptorFingerprintProcessor.compute_descriptors)
        df_cleaned = pd.concat([df_cleaned, pd.DataFrame(descriptors.tolist())], axis=1)

        # Generate fingerprints
        df_cleaned['fingerprints'] = DescriptorFingerprintProcessor.generate_rdkit_fingerprints(df_cleaned['canonical_smiles'])
        fingerprints = df_cleaned['fingerprints'].tolist()

        # Compute average Tanimoto similarity
        df_cleaned['avg_tanimoto'] = DescriptorFingerprintProcessor.compute_avg_tanimoto_in_batches(
            fingerprints=fingerprints, batch_size=5000
        )

        # Perform Butina clustering
        butina_clusters = ClusteringAndOutlierDetection.butina_clustering(fingerprints, cutoff=self.butina_cutoff)
        print(f"Number of clusters from Butina: {len(butina_clusters)}")

        cluster_labels = np.zeros(len(df_cleaned), dtype=int) - 1  # Initialize with -1 (unclustered)
        for cluster_idx, cluster in enumerate(butina_clusters):
            for molecule_idx in cluster:
                cluster_labels[molecule_idx] = cluster_idx
        df_cleaned['butina_cluster'] = cluster_labels
        print(f"Butina clustering results:\n{df_cleaned['butina_cluster'].value_counts()}")  # Debugging checkpoint

        # Prepare features
        numerical_features = FeatureProcessor.prepare_features(df_cleaned)

        # Reduce dimensions with UMAP
        reduced_features = FeatureProcessor.apply_umap(numerical_features, n_neighbors=self.n_neighbors, min_dist=self.min_dist)

        # HAC on UMAP-reduced features
        hac_labels = ClusteringAndOutlierDetection.apply_hac(reduced_features, n_clusters=None, distance_threshold=1.5)
        Visualizer.visualize_clusters(reduced_features, hac_labels, f"{self.dataset_name} Clusters (Butina + HAC)")

        # Validation Scores for Butina + HAC
        # Evaluate Clustering
        evaluation = Evaluation()
        evaluation.evaluate(umap_features, labels)

        return df_cleaned

    def process_and_cluster(self):
        """Full pipeline to process, detect outliers, cluster with Butina + HAC and HAC separately."""
        # Load data
        df = pd.read_csv(self.data_path, sep="\t")
        print(f"{self.dataset_name} initial data shape: {df.shape}")  # Debugging checkpoint
        df['source'] = self.source_type
        print(f"{self.dataset_name} source column values:\n{df['source'].value_counts()}")  # Debugging checkpoint

        # Compute descriptors
        descriptors = df['canonical_smiles'].apply(DescriptorFingerprintProcessor.compute_descriptors)
        df = pd.concat([df, pd.DataFrame(descriptors.tolist())], axis=1)

        # Generate fingerprints
        df['fingerprints'] = DescriptorFingerprintProcessor.generate_rdkit_fingerprints(df['canonical_smiles'])
        fingerprints = df['fingerprints'].tolist()

        # Compute average Tanimoto similarity
        # Use batch processing to handle large datasets
        df['avg_tanimoto'] = DescriptorFingerprintProcessor.compute_avg_tanimoto_in_batches(
            fingerprints=df['fingerprints'].tolist(), 
            batch_size=5000
        )

        # Butina Clustering
        butina_clusters = ClusteringAndOutlierDetection.butina_clustering(fingerprints, cutoff=self.butina_cutoff)
        print(f"Number of clusters from Butina: {len(butina_clusters)}")

        cluster_labels = np.zeros(len(df), dtype=int) - 1  # Initialize with -1 (unclustered)
        for cluster_idx, cluster in enumerate(butina_clusters):
            for molecule_idx in cluster:
                cluster_labels[molecule_idx] = cluster_idx
        df['butina_cluster'] = cluster_labels
        print(f"Butina clustering results:\n{df['butina_cluster'].value_counts()}")  # Debugging checkpoint

        # Prepare features
        numerical_features = FeatureProcessor.prepare_features(df)

        # Reduce dimensions with UMAP
        reduced_features = FeatureProcessor.apply_umap(numerical_features, n_neighbors=self.n_neighbors, min_dist=self.min_dist)

        # HAC on UMAP-reduced features
        hac_labels = ClusteringAndOutlierDetection.apply_hac(reduced_features, n_clusters=None, distance_threshold=1.5)
        Visualizer.visualize_clusters(reduced_features, hac_labels, f"{self.dataset_name} Clusters (Butina + HAC)")

        # Validation Scores for Butina + HAC
        # Evaluate Clustering
        evaluation = Evaluation()
        evaluation.evaluate(umap_features, labels)

        # HAC with Outlier Detection
        isolation_outliers = ClusteringAndOutlierDetection.detect_outliers_isolation(numerical_features)
        elliptic_outliers = ClusteringAndOutlierDetection.detect_outliers_elliptic(numerical_features)
        df = ClusteringAndOutlierDetection.compare_outliers(df, isolation_outliers, elliptic_outliers)

        inliers_df = df[df["combined_outlier"] == "Inlier"]
        inlier_features = FeatureProcessor.prepare_features(inliers_df)

        reduced_features_hac = FeatureProcessor.apply_umap(inlier_features, n_neighbors=self.n_neighbors, min_dist=self.min_dist)
        hac_labels_outlier = ClusteringAndOutlierDetection.apply_hac(reduced_features_hac, n_clusters=None, distance_threshold=1.5)
        Visualizer.visualize_clusters(reduced_features_hac, hac_labels_outlier, f"{self.dataset_name} Clusters (HAC with Outliers)")

        # Plot Outliers
        outlier_labels = df["combined_outlier"].values
        Visualizer.plot_outliers(numerical_features, outlier_labels, f"{self.dataset_name} Outlier Detection")

        # Validation Scores for HAC with Outliers
        # Evaluate Clustering
        evaluation = Evaluation()
        evaluation.evaluate(umap_features, labels)

        return df
