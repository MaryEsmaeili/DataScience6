import pandas as pd

class DataLoader:
    def __init__(self, synthetic_path, natural_path):
        self.synthetic_path = synthetic_path
        self.natural_path = natural_path

    def load_data(self):
        try:
            # Correct column names
            columns = ["compound_name", "molecular_formula", "canonical_smiles"]

            # Read datasets with explicit column names
            synthetic_df = pd.read_csv(self.synthetic_path, sep="\t", names=columns, header=None, engine='python')
            natural_df = pd.read_csv(self.natural_path, sep="\t", names=columns, header=None, engine='python')

            # Add source column
            synthetic_df['source'] = 1
            natural_df['source'] = 0

            # Combine datasets
            combined_df = pd.concat([synthetic_df, natural_df], ignore_index=True)
            return combined_df
        except Exception as e:
            print(f"Error loading or processing files: {e}")
            return None
