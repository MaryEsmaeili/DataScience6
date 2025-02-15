import logging
import pandas as pd
from rdkit import Chem

class DataCleaner:
    def __init__(self, log_file="cleaning.log"):
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )

    @staticmethod
    def is_utf8(cell):
        """Check if a string can be encoded as UTF-8."""
        try:
            cell.encode("utf-8")
            return True
        except Exception:
            return False

    @staticmethod
    def is_valid_smiles(smiles):
        """Check if a SMILES string is valid."""
        try:
            return Chem.MolFromSmiles(smiles) is not None
        except Exception:
            return False

    def clean(self, df):
        """
        Cleans the dataset by:
        - Removing duplicate rows
        - Dropping missing critical values (canonical_smiles)
        - Validating SMILES strings
        - Ensuring UTF-8 encoding for text fields
        - Returning only necessary columns

        Args:
            df (pd.DataFrame): Input DataFrame with at least `compound_name`, `molecular_formula`, and `canonical_smiles`.

        Returns:
            pd.DataFrame: Cleaned DataFrame.
        """
        original_count = len(df)

        # Remove duplicates
        df = df.drop_duplicates()

        # Drop rows with missing SMILES
        df = df.dropna(subset=["canonical_smiles"])

        # Standardize column names
        df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

        # Validate SMILES and UTF-8 encoding
        df = df[
            df["canonical_smiles"].apply(self.is_valid_smiles) &
            df["compound_name"].apply(self.is_utf8) &
            df["molecular_formula"].apply(self.is_utf8) &
            df["canonical_smiles"].apply(self.is_utf8)
        ]

        # Select relevant columns
        clean_df = df[["compound_name", "molecular_formula", "canonical_smiles"]].copy()

        # Logging and reporting
        cleaned_count = len(clean_df)
        deleted_count = original_count - cleaned_count
        logging.info(f"Cleaning complete: {cleaned_count} valid rows retained, {deleted_count} rows removed.")

        return clean_df
