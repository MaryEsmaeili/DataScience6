
import pandas as pd
from modules.combine_data_processor import CombinedDatasetProcessor
from modules.natural_substitutes_saver import NaturalSubstituteSaver
from modules.pipeline import MolecularPipeline
from modules.substitute_analyzer import SubstituteAnalyzer
from modules.visualizer import Visualizer
from modules.mix_cluster_identifier import MixedClusterVisualizer

if __name__ == "__main__":
    # Process synthetic and natural datasets
    synthetic_pipeline = MolecularPipeline(
        "/data/non_natural.txt",
        "Synthetic Dataset",
        "Synthetic"
    )
    synthetic_df = synthetic_pipeline.process_and_cluster()

    natural_pipeline = MolecularPipeline(
        "/data/subset_natural.txt",
        "Natural Dataset",
        "Natural"
    )
    natural_df = natural_pipeline.process_and_cluster()

    # Combine datasets and save
    combined_df = pd.concat([synthetic_df, natural_df], ignore_index=True)
    combined_dataset_path = "/homes/mesmaeili/Documents/DS6/data/combined_dataset.csv"
    combined_df.to_csv(combined_dataset_path, index=False)
    print(f"Combined dataset saved to '{combined_dataset_path}'.")

    # Plot histogram of average Tanimoto distances
    Visualizer.plot_tanimoto_histogram(synthetic_df, natural_df)

    # Process the combined dataset
    print("\nProcessing Combined Dataset...")
    combined_df, combined_butina_labels, combined_hac_labels, combined_reduced_features = CombinedDatasetProcessor.process_combined_dataset(
        combined_df, butina_cutoff=0.5, n_neighbors=15, min_dist=0.01
    )

    # Identify and visualize mixed clusters
    MixedClusterVisualizer.find_and_visualize_mixed_clusters(
        combined_df, combined_reduced_features, cluster_column="butina_cluster", title="Mixed Clusters (Butina)"
    )

    # Save natural compounds in mixed clusters
    output_file = "natural_substitutes.csv"
    NaturalSubstituteSaver.save_natural_substitutes_to_csv(
        combined_df,
        cluster_column="butina_cluster",
        source_column="source",
        output_file=output_file
    )
    print(f"Natural substitutes saved to '{output_file}'.")

    # Find common natural substitutes
    substitutes_df = SubstituteAnalyzer.find_common_natural_substitutes(
        combined_df,
        cluster_column="butina_cluster",
        similarity_threshold=0.2
    )

    # Save and analyze common natural substitutes
    substitutes_output_file = "natural_synthetic_substitutes.csv"
    if not substitutes_df.empty:
        substitutes_df.to_csv(substitutes_output_file, index=False)
        print(f"Results saved to '{substitutes_output_file}'.")

        # Plot property differences and display top substitutes
        SubstituteAnalyzer.plot_property_differences(substitutes_df)
        SubstituteAnalyzer.display_top_substitutes(substitutes_df, top_n=5)
    else:
        print("No substitutes found. Try lowering the similarity threshold.")
