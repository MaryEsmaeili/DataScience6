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
            logging.warning(f"Non-UTF-8 detected: {cell}")
            return False

    @staticmethod
    def detect_invalid_smiles(smiles):
        """Check if a SMILES string is valid."""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                logging.warning(f"Invalid SMILES: {smiles}")
                return False
            return True
        except Exception as e:
            logging.error(f"Error parsing SMILES {smiles}: {e}")
            return False

    def clean(self, df):
        """
        Clean the dataset by validating SMILES strings and UTF-8 encoding.

        Args:
            df (pd.DataFrame): Input DataFrame with `compound_name`, `molecular_formula`, `canonical_smiles`.

        Returns:
            pd.DataFrame: Cleaned DataFrame containing only valid and UTF-8 encoded rows.
        """
        logging.info("Starting data validation...")

        # Validate SMILES strings
        df["valid"] = df["canonical_smiles"].apply(self.detect_invalid_smiles)

        # Validate UTF-8 encoding
        df["is_utf8_name"] = df["compound_name"].apply(self.is_utf8)
        df["is_utf8_formula"] = df["molecular_formula"].apply(self.is_utf8)
        df["is_utf8_smiles"] = df["canonical_smiles"].apply(self.is_utf8)

        # Filter valid rows
        logging.info("Filtering valid and UTF-8 encoded rows...")
        df_cleaned = df[
            (df["valid"]) &
            (df["is_utf8_name"]) &
            (df["is_utf8_formula"]) &
            (df["is_utf8_smiles"])
        ]

        logging.info(f"Finished cleaning. {len(df_cleaned)} valid rows retained out of {len(df)} total rows.")
        return df_cleaned
