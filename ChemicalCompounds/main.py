from modules.data_loader import DataLoader
from modules.data_cleaner import DataCleaner
from modules.feature_generator import FeatureGenerator
from modules.clustering import Clustering
from modules.evaluation import Evaluation

# Define file paths
synthetic_path = "/homes/mesmaeili/Documents/DS6/data/non_natural_products.txt"
natural_path = "/homes/mesmaeili/Documents/DS6/data/subset_natural_products.txt"

# Load Data
data_loader = DataLoader(synthetic_path, natural_path)
df = data_loader.load_data()

# Validate & Clean Data
cleaner = DataCleaner()
cleaned_df = cleaner.clean(df)

# Generate Features
feature_gen = FeatureGenerator()
features_df, fingerprints_df = feature_gen.process_features(cleaned_df)

# Scale, Apply UMAP, and Cluster
clustering = Clustering(n_clusters=6, n_neighbors=15, min_dist=0.1, n_components=2)
umap_features = clustering.scale_and_umap(features_df)
labels = clustering.perform_clustering(umap_features)
clustering.visualize_clusters(umap_features, labels)

# Evaluate Clustering
evaluation = Evaluation()
evaluation.evaluate(umap_features, labels)
