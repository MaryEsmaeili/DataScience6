from sklearn.discriminant_analysis import StandardScaler
import umap

class FeatureProcessor:
    @staticmethod
    def prepare_features(df):
        """Prepare and scale features for clustering."""
        features = df[['Molecular Weight',
                'Heavy atoms',
                'logP',
                'Refractivity',
                'TPSA',
                'fraction csp3',
                'H-Bond Donors',
                'H-Bond Acceptors',
                'Rotatable bonds',
                'Num rings',
                'Aromatic Rings',
                'Aliphatic_rings',
                'Stereo centers',
                'Charge',
                'Valence electrons']].fillna(0).values
        scaler = StandardScaler()
        return scaler.fit_transform(features)

    @staticmethod
    def apply_umap(features, n_neighbors=20, min_dist=0.01, n_components=2, random_state=42):
        """Apply UMAP for dimensionality reduction."""
        reducer = umap.UMAP(
            n_neighbors=n_neighbors, 
            min_dist=min_dist, 
            n_components=n_components, 
            random_state=random_state
        )
        return reducer.fit_transform(features)