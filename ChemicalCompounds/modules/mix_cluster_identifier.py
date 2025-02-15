class MixedClusterIdentifier:
    @staticmethod
    def identify_mixed_clusters(combined_df, cluster_column):
        """Identify clusters containing both synthetic and natural compounds."""
        mixed_clusters = []
        grouped = combined_df.groupby(cluster_column)
        for cluster, group in grouped:
            sources = group['source'].unique()
            print(f"Cluster {cluster}: Sources: {sources}")  # Debugging checkpoint
            if len(sources) > 1:  # More than one source (e.g., Synthetic and Natural)
                mixed_clusters.append(cluster)
        print(f"Mixed clusters: {mixed_clusters}")
        return mixed_clusters