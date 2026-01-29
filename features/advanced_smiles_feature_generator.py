import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.decomposition import TruncatedSVD


class AdvancedSmilesFeatureGenerator:
    """
    Generates advanced SMILES features using Morgan fingerprints
    reduced via TruncatedSVD.

    NOTE:
    - This version fits SVD internally.
    - Use ONLY on training data or full dataset (no CV safety).
    """

    def __init__(
        self,
        smiles_col: str = "SMILES",
        radius: int = 2,
        n_bits: int = 2048,
        n_components: int = 100,
        random_state: int = 42,
    ):
        self.smiles_col = smiles_col
        self.radius = radius
        self.n_bits = n_bits
        self.n_components = n_components
        self.random_state = random_state

    def _smiles_to_fp(self, smiles: str) -> np.ndarray:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(self.n_bits, dtype=np.int8)

        fp = AllChem.GetMorganFingerprintAsBitVect(
            mol,
            radius=self.radius,
            nBits=self.n_bits,
        )
        return np.asarray(fp, dtype=np.int8)

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        # ---- Step 1: Fingerprint matrix ----
        X_fp = np.vstack(
            df[self.smiles_col].apply(self._smiles_to_fp).values
        )

        # ---- Step 2: TruncatedSVD ----
        svd = TruncatedSVD(
            n_components=self.n_components,
            random_state=self.random_state,
        )
        X_reduced = svd.fit_transform(X_fp)

        # ---- Step 3: Return DataFrame ----
        columns = [f"fp_svd_{i+1}" for i in range(self.n_components)]

        return pd.DataFrame(
            X_reduced,
            columns=columns,
            index=df.index,
        )
