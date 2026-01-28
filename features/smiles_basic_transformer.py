import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import Descriptors

from sklearn.base import BaseEstimator, TransformerMixin


class BasicSmilesFeatures(BaseEstimator, TransformerMixin):
    """
    Transformer that takes a full DataFrame, extracts basic molecular
    features from a SMILES column, and returns a NEW DataFrame with
    those features appended.
    """

    def __init__(self, smiles_col="SMILES"):
        self.smiles_col = smiles_col

    def fit(self, df, y=None):
        # No fitting required (rule-based transformer)
        return self

    def transform(self, df):
        """
        Parameters
        ----------
        df : pd.DataFrame
            Original dataframe containing a SMILES column

        Returns
        -------
        pd.DataFrame
            New dataframe with basic molecular features added
        """

        # Work on a copy to avoid modifying the original dataframe
        df_out = df.copy()

        def generate_basic_features(smiles):
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return {
                    "mol_weight": np.nan,
                    "num_atoms": np.nan,
                    "num_rings": np.nan,
                    "num_C": np.nan,
                    "num_N": np.nan,
                    "num_O": np.nan,
                    "num_S": np.nan,
                }

            return {
                # Molecular size
                "mol_weight": Descriptors.MolWt(mol),

                # Number of explicit (non-hydrogen) atoms
                "num_atoms": mol.GetNumAtoms(),

                # Number of rings (topological cycles)
                "num_rings": Chem.GetSSSR(mol),

                # Elemental composition
                "num_C": sum(a.GetSymbol() == "C" for a in mol.GetAtoms()),
                "num_N": sum(a.GetSymbol() == "N" for a in mol.GetAtoms()),
                "num_O": sum(a.GetSymbol() == "O" for a in mol.GetAtoms()),
                "num_S": sum(a.GetSymbol() == "S" for a in mol.GetAtoms()),
            }

        # Generate features from SMILES
        features = df_out[self.smiles_col].apply(generate_basic_features)
        features_df = features.apply(pd.Series)

        # Append features to original dataframe
        df_out = pd.concat(
            [df_out.reset_index(drop=True), features_df.reset_index(drop=True)],
            axis=1
        )

        return df_out

# Example usage:
# df = pd.DataFrame({"SMILES": ["CCO", "c1ccccc1", "CC(=O)O"]})
# transformer = BasicSmilesFeatures(smiles_col="SMILES")
# df_transformed = transformer.fit_transform(df)