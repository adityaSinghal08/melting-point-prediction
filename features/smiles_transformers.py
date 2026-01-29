import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from features.basic_smiles_feature_generator import BasicSmilesFeatureGenerator
from features.intermediate_smiles_feature_generator import IntermediateSmilesFeatureGenerator
from features.advanced_smiles_feature_generator import AdvancedSmilesFeatureGenerator


class BasicSmilesTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, smiles_col="SMILES"):
        self.smiles_col = smiles_col
        self.generator = BasicSmilesFeatureGenerator(smiles_col)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return self.generator.generate(X)


class IntermediateSmilesTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, smiles_col="SMILES"):
        self.smiles_col = smiles_col
        self.generator = IntermediateSmilesFeatureGenerator(smiles_col)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return self.generator.generate(X)


class AdvancedSmilesTransformer(BaseEstimator, TransformerMixin):
    """
    NOTE:
    - Uses DF-only AdvancedSmilesFeatureGenerator
    - NOT CV-safe (fits SVD internally)
    - OK for experimentation
    """
    def __init__(self, smiles_col="SMILES"):
        self.smiles_col = smiles_col
        self.generator = AdvancedSmilesFeatureGenerator(smiles_col)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return self.generator.generate(X)


class DropSmilesTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, smiles_col="SMILES"):
        self.smiles_col = smiles_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop(columns=[self.smiles_col])
