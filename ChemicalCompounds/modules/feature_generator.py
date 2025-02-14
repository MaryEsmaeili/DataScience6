import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.DataStructs import ConvertToNumpyArray

class FeatureGenerator:
    @staticmethod
    def calculate_properties(smiles):
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            return None

        return {
            "Molecular Weight": Descriptors.MolWt(mol),
            "logP": Descriptors.MolLogP(mol),
            "TPSA": rdMolDescriptors.CalcTPSA(mol),
            "H-Bond Donors": Descriptors.NumHDonors(mol),
            "H-Bond Acceptors": Descriptors.NumHAcceptors(mol),
            "Rotatable bonds": Descriptors.NumRotatableBonds(mol),
        }

    @staticmethod
    def get_fingerprint(smiles, radius=2, n_bits=1024):
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            return np.array(rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits))
        return np.zeros(n_bits, dtype=int)

    def process_features(self, dataframe):
        dataframe['properties'] = dataframe['canonical_smiles'].apply(self.calculate_properties)
        properties_df = pd.DataFrame(dataframe['properties'].tolist())

        dataframe['fingerprints'] = dataframe['canonical_smiles'].apply(self.get_fingerprint)
        fingerprints_df = pd.DataFrame(dataframe['fingerprints'].tolist())

        return properties_df, fingerprints_df
