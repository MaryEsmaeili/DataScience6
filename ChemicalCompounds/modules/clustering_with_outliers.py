from sklearn.cluster import AgglomerativeClustering
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from rdkit.ML.Cluster import Butina
from modules.descriptor_fingerprint import DescriptorFingerprintProcessor


class ClusteringAndOutlierDetection:
    
    @staticmethod
    def butina_clustering(fingerprints, cutoff=0.5):
        """Perform Butina clustering on molecular fingerprints."""
        distances = DescriptorFingerprintProcessor.compute_tanimoto_distances(fingerprints)
        n_fps = len(fingerprints)
        upper_triangle_distances = []
        for i in range(n_fps):
            for j in range(i + 1, n_fps):
                upper_triangle_distances.append(1 - distances[i, j])
        clusters = Butina.ClusterData(upper_triangle_distances, n_fps, cutoff, isDistData=True)
        return clusters

    @staticmethod
    def apply_hac(features, n_clusters=None, distance_threshold=1.0):
        """Apply Hierarchical Agglomerative Clustering (HAC)."""
        hac = AgglomerativeClustering(
            n_clusters=n_clusters, 
            distance_threshold=distance_threshold, 
            linkage='average'
        )
        return hac.fit_predict(features)

    @staticmethod
    def detect_outliers_isolation(features, contamination=0.05):
        """Detect outliers using Isolation Forest."""
        model = IsolationForest(contamination=contamination, random_state=42)
        return model.fit_predict(features)

    @staticmethod
    def detect_outliers_elliptic(features, contamination=0.05):
        """Detect outliers using Elliptic Envelope."""
        model = EllipticEnvelope(contamination=contamination, random_state=42)
        return model.fit_predict(features)

    @staticmethod
    def compare_outliers(df, outlier_isolation, outlier_elliptic):
        """Combine outlier results from Isolation Forest and Elliptic Envelope."""
        combined_status = []
        for iso, ell in zip(outlier_isolation, outlier_elliptic):
            if iso == -1 and ell == -1:
                combined_status.append("Both")
            elif iso == -1:
                combined_status.append("Isolation Only")
            elif ell == -1:
                combined_status.append("Elliptic Only")
            else:
                combined_status.append("Inlier")
        df["combined_outlier"] = combined_status
        return df