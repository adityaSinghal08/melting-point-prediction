import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import Descriptors

from sklearn.base import BaseEstimator, TransformerMixin


class BasicSmilesFeatures(BaseEstimator, TransformerMixin):
    """
    sklearn-compatible transformer that converts SMILES strings
    into basic molecular features.
    """

    def __init__(self, smiles_col="SMILES"):
        self.smiles_col = smiles_col

    def fit(self, X, y=None):
        """
        No fitting required: this transformer is rule-based.
        """
        return self

    def transform(self, X):
        """
        Converts SMILES strings into numeric molecular features.

        Parameters
        ----------
        X : pd.DataFrame
            DataFrame containing a SMILES column

        Returns
        -------
        pd.DataFrame
            DataFrame of basic molecular features
        """

        def featurize(smiles):
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

        features = X[self.smiles_col].apply(featurize)
        return features.apply(pd.Series)