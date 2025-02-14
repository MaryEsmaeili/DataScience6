from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

class Evaluation:
    @staticmethod
    def evaluate(data, labels):
        for method, label in labels.items():
            if len(set(label)) > 1:
                sil_score = silhouette_score(data, label)
                db_score = davies_bouldin_score(data, label)
                ch_score = calinski_harabasz_score(data, label)
                print(f"{method} - Silhouette Score: {sil_score:.2f}")
                print(f"{method} - Davies-Bouldin Score: {db_score:.2f}")
                print(f"{method} - Calinski-Harabasz Index: {ch_score:.2f}")
            else:
                print(f"{method} - Only one cluster found.")
