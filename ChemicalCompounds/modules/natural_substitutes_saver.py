class NaturalSubstituteSaver:
    @staticmethod
    def save_natural_substitutes_to_csv(combined_df, cluster_column, source_column, output_file):
        """
        Identify natural compounds in mixed clusters and save them as a CSV file.

        Parameters:
        - combined_df: pd.DataFrame, combined dataset with both synthetic and natural compounds.
        - cluster_column: str, the column name for cluster labels.
        - source_column: str, the column name for source type ('Synthetic' or 'Natural').
        - output_file: str, the file name to save the results as a CSV file.

        Returns:
        - None
        """
        # Identify mixed clusters containing both synthetic and natural compounds
        mixed_clusters = []
        grouped = combined_df.groupby(cluster_column)
        for cluster, group in grouped:
            sources = group[source_column].unique()
            if len(sources) > 1:  # More than one source (e.g., Synthetic and Natural)
                mixed_clusters.append(cluster)

        print(f"Mixed Clusters (Containing Both Synthetic and Natural): {mixed_clusters}")

        # Filter for natural compounds in mixed clusters
        natural_substitutes = combined_df[
            (combined_df[cluster_column].isin(mixed_clusters)) & (combined_df[source_column] == 'Natural')
        ]

        print(f"Number of Natural Compounds in Mixed Clusters: {len(natural_substitutes)}")

        # Save to CSV
        natural_substitutes.to_csv(output_file, index=False)
        print(f"Natural substitutes saved to {output_file}")

