import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors


class BasicSmilesFeatureGenerator:
    """
    Generates basic molecular features from a SMILES column
    and returns a DataFrame of new features.
    """

    def __init__(self, smiles_col="SMILES"):
        self.smiles_col = smiles_col

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing a SMILES column

        Returns
        -------
        pd.DataFrame
            DataFrame with basic molecular features
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
                "mol_weight": Descriptors.MolWt(mol),
                "num_atoms": mol.GetNumAtoms(),
                "num_rings": len(Chem.GetSSSR(mol)),
                "num_C": sum(a.GetSymbol() == "C" for a in mol.GetAtoms()),
                "num_N": sum(a.GetSymbol() == "N" for a in mol.GetAtoms()),
                "num_O": sum(a.GetSymbol() == "O" for a in mol.GetAtoms()),
                "num_S": sum(a.GetSymbol() == "S" for a in mol.GetAtoms()),
            }

        features = df[self.smiles_col].apply(featurize)
        return features.apply(pd.Series)
