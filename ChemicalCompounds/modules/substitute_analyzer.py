import math
from rdkit.Chem import DataStructs
from matplotlib import pyplot as plt
from modules.mix_cluster_identifier import MixedClusterIdentifier


class SubstituteAnalyzer:
    @staticmethod
    def find_common_natural_substitutes(combined_df, cluster_column, similarity_threshold=0.8):
        """
        Identify natural compounds that can substitute synthetic compounds in mixed clusters.

        Parameters:
        - combined_df: pd.DataFrame, combined dataset with both synthetic and natural compounds.
        - cluster_column: str, the column name for cluster labels.
        - similarity_threshold: float, Tanimoto similarity threshold for substitution.

        Returns:
        - substitutes_df: pd.DataFrame, details of natural compounds and why they can substitute.
        """
        # Identify mixed clusters containing both synthetic and natural compounds
        mixed_clusters = MixedClusterIdentifier.identify_mixed_clusters(combined_df, cluster_column)

        # Filter natural and synthetic compounds in mixed clusters
        natural_in_mixed = combined_df[
            (combined_df[cluster_column].isin(mixed_clusters)) & (combined_df["source"] == "Natural")
        ]
        synthetic_in_mixed = combined_df[
            (combined_df[cluster_column].isin(mixed_clusters)) & (combined_df["source"] == "Synthetic")
        ]

        print(f"Natural compounds in mixed clusters: {len(natural_in_mixed)}")
        print(f"Synthetic compounds in mixed clusters: {len(synthetic_in_mixed)}")

        # Pairwise comparison between natural and synthetic compounds within mixed clusters
        substitutes = []
        for cluster in mixed_clusters:
            natural_cluster = natural_in_mixed[natural_in_mixed[cluster_column] == cluster]
            synthetic_cluster = synthetic_in_mixed[synthetic_in_mixed[cluster_column] == cluster]

            for _, nat_row in natural_cluster.iterrows():
                for _, syn_row in synthetic_cluster.iterrows():
                    if nat_row["fingerprints"] and syn_row["fingerprints"]:
                        # Compute Tanimoto similarity between fingerprints
                        tanimoto_sim = DataStructs.TanimotoSimilarity(
                            nat_row["fingerprints"], syn_row["fingerprints"]
                        )
                        if tanimoto_sim >= similarity_threshold:
                            # Compare molecular properties
                            property_diff = {
                                "MolWt_Diff": abs(nat_row.get("MolWt", 0) - syn_row.get("MolWt", 0)),
                                "LogP_Diff": abs(nat_row.get("LogP", 0) - syn_row.get("LogP", 0)),
                                "TPSA_Diff": abs(nat_row.get("TPSA", 0) - syn_row.get("TPSA", 0)),
                                "HDonors_Diff": abs(nat_row.get("NumHDonors", 0) - syn_row.get("NumHDonors", 0)),
                                "HAcceptors_Diff": abs(nat_row.get("NumHAcceptors", 0) - syn_row.get("NumHAcceptors", 0)),
                                "Hatom_Diff": abs(nat_row.get("Hatom_Diff", 0) - syn_row.get("Hatom_Diff", 0)),
                                "Refractivity_Diff": abs(nat_row.get("Refractivity_Diff", 0) - syn_row.get("Refractivity_Diff", 0)),
                                "fractionCSP3_Diff": abs(nat_row.get("fractionCSP3_Diff", 0) - syn_row.get("fractionCSP3_Diff", 0)),
                                "RotatableBonds_Diff": abs(nat_row.get("RotatableBonds_Diff", 0) - syn_row.get("RotatableBonds_Diff", 0)),
                                "NumRings_Diff": abs(nat_row.get("NumRings_Diff", 0) - syn_row.get("NumRings_Diff", 0)),
                                "AromaticRings_Diff": abs(nat_row.get("AromaticRings_Diff", 0) - syn_row.get("AromaticRings_Diff", 0)),
                                "Aliphatic_rings_Diff": abs(nat_row.get("Aliphatic_rings_Diff", 0) - syn_row.get("Aliphatic_rings_Diff", 0)),
                                "StereoCenters_Diff": abs(nat_row.get("StereoCenters_Diff", 0) - syn_row.get("StereoCenters_Diff", 0)),
                                "Charge_Diff": abs(nat_row.get("Charge_Diff", 0) - syn_row.get("Charge_Diff", 0)),
                                "ValenceElectrons_Diff": abs(nat_row.get("ValenceElectrons_Diff", 0) - syn_row.get("ValenceElectrons_Diff", 0)),                                            }
                            substitutes.append({
                                "Natural_Compound": nat_row["compound_name"],
                                "Synthetic_Compound": syn_row["compound_name"],
                                "Cluster": cluster,
                                "Tanimoto_Similarity": tanimoto_sim,
                                **property_diff
                            })
                            print(f"Match found: {nat_row['compound_name']} -> {syn_row['compound_name']} (Tanimoto: {tanimoto_sim:.2f})")

        substitutes_df = pd.DataFrame(substitutes)
        print(f"Total substitutes found: {len(substitutes_df)}")
        return substitutes_df

    @staticmethod
    def plot_property_differences(substitutes_df):
        """Plot histograms of property differences between natural and synthetic compounds."""
        properties = [
            "MolWt_Diff", 
            "LogP_Diff", 
            "TPSA_Diff",
            "HDonors_Diff",
            "HAcceptors_Diff",
            "Hatom_Diff",
            "Refractivity_Diff", 
            "fractionCSP3_Diff", 
            "RotatableBonds_Diff", 
            "NumRings_Diff", 
            "AromaticRings_Diff",
            "Aliphatic_rings_Diff",
            "StereoCenters_Diff",
            "Charge_Diff",
            "ValenceElectrons_Diff"
        ]
        
        num_props = len(properties)
        cols = 3  # Number of columns per row
        rows = math.ceil(num_props / cols)  # Dynamically calculate the number of rows
        
        plt.figure(figsize=(cols * 5, rows * 4))  # Adjust figure size dynamically
        for i, prop in enumerate(properties, start=1):
            plt.subplot(rows, cols, i)  # Dynamically adjust the subplot grid
            substitutes_df[prop].hist(bins=20, alpha=0.7, color='blue', edgecolor='black')
            plt.title(f"Distribution of {prop}")
            plt.xlabel(prop)
            plt.ylabel("Frequency")
        
        plt.tight_layout()
        plt.suptitle("Differences in Molecular Properties (Natural vs Synthetic)", y=1.02, fontsize=16)
        plt.show()


    @staticmethod
    def display_top_substitutes(substitutes_df, top_n=5):
        """Display the top N natural substitutes sorted by Tanimoto similarity."""
        top_substitutes = substitutes_df.sort_values(by="Tanimoto_Similarity", ascending=False).head(top_n)
        print("Top Substitutes:")
        print(top_substitutes)