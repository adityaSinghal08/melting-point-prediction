import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors


class IntermediateSmilesFeatureGenerator:
    """
    Generates chemistry-aware intermediate features from SMILES.
    Assumes basic atomic/count features already exist.
    """

    def __init__(self, smiles_col="SMILES"):
        self.smiles_col = smiles_col

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:

        def featurize(smiles):
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return {}

            atoms = mol.GetAtoms()
            heavy_atoms = mol.GetNumHeavyAtoms()
            num_atoms = mol.GetNumAtoms()

            num_aromatic_atoms = sum(a.GetIsAromatic() for a in atoms)
            num_hetero_atoms = sum(a.GetSymbol() not in ["C", "H"] for a in atoms)
            num_halogen_atoms = sum(
                a.GetSymbol() in ["F", "Cl", "Br", "I"] for a in atoms
            )

            num_aromatic_rings = rdMolDescriptors.CalcNumAromaticRings(mol)
            num_rot_bonds = Descriptors.NumRotatableBonds(mol)

            return {
                # ---- Polarity & Interactions ----
                "logp": Descriptors.MolLogP(mol),
                "tpsa": Descriptors.TPSA(mol),
                "h_donors": Descriptors.NumHDonors(mol),
                "h_acceptors": Descriptors.NumHAcceptors(mol),
                "nhoh_count": Descriptors.NHOHCount(mol),

                # ---- Aromaticity & Rings ----
                "num_aromatic_rings": num_aromatic_rings,
                "num_aromatic_atoms": num_aromatic_atoms,

                # ---- Flexibility ----
                "num_rotatable_bonds": num_rot_bonds,
                "fraction_csp3": Descriptors.FractionCSP3(mol),

                # ---- Composition ----
                "num_hetero_atoms": num_hetero_atoms,
                "num_halogen_atoms": num_halogen_atoms,

                # ---- Topology / Shape ----
                "bertz_ct": Descriptors.BertzCT(mol),
                "balaban_j": Descriptors.BalabanJ(mol),
                "hall_kier_alpha": Descriptors.HallKierAlpha(mol),

                # ---- Ratios (this is where models smile) ----
                "aromatic_atom_ratio": num_aromatic_atoms / num_atoms if num_atoms else 0,
                "rotatable_bond_ratio": num_rot_bonds / heavy_atoms if heavy_atoms else 0,
                "hetero_atom_ratio": num_hetero_atoms / heavy_atoms if heavy_atoms else 0,
                "aromatic_ring_ratio": num_aromatic_rings / heavy_atoms if heavy_atoms else 0,
            }

        features = df[self.smiles_col].apply(featurize)
        return pd.DataFrame(list(features))
