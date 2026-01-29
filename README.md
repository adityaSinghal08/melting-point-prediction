# 🔥 Melting Point Prediction using Machine Learning

This project focuses on predicting the **melting point (Tm in Kelvin)** of organic compounds using **molecular descriptors** and **SMILES-based features**. It is built as part of a **Kaggle regression competition** involving chemical property prediction. The repository explores **both traditional descriptor-based models** and **SMILES-driven feature engineering pipelines**.

---

## 📌 Problem Statement

**Given:**
- **SMILES strings** representing organic molecules
- **Precomputed molecular descriptors** and **custom SMILES features**

**Predict:**
- **Melting Point (Tm)** in **Kelvin**

This is a **supervised regression problem** with high-dimensional tabular data.

---

## 📂 Dataset Description

The dataset consists of:

| Column | Description |
|--------|-------------|
| `id` | Unique compound identifier |
| `SMILES` | Chemical structure representation |
| `Tm` | Target variable (melting point in Kelvin) |
| `Group 1 ... Group N` | Molecular descriptors |

- `Tm` is available only in the training set
- `train_improved.csv` contains additional preprocessing / feature refinements

---

## 🧠 Project Approach

### 🔹 Data Exploration
- Distribution analysis of melting points
- Descriptor correlation analysis
- SMILES structure exploration

### 🔹 Feature Engineering
- Descriptor-based features
- Custom SMILES-based features:
  - **Basic features**: Molecular weight, atom counts, ring counts
  - **Intermediate features**: Structural patterns and functional groups
  - **Advanced features**: Complex molecular characteristics

### 🔹 Modeling Strategies
Two parallel modeling pipelines were explored:

1. **Without SMILES** - Uses only numerical molecular descriptors
2. **With SMILES** - Incorporates engineered SMILES features

---

## 🧪 Models Implemented

### 📘 Models (Without SMILES)
- Linear Regression
- Random Forest Regressor
- XGBoost Regressor
- LightGBM Regressor

### 🧬 Models (Using SMILES)
- XGBoost with SMILES features
- LightGBM with SMILES features

---

## ⚙️ Tech Stack

- **Python**
- **Pandas, NumPy**
- **Scikit-learn**
- **XGBoost**
- **LightGBM**
- **RDKit** *(for SMILES processing)*
- **Matplotlib / Seaborn**
- **Jupyter Notebook**

---

## 📁 Project Structure
```
melting-point-prediction/
├── data/
│   ├── train.csv                    # Training dataset
│   ├── train_improved.csv           # Preprocessed training data
│   └── test.csv                     # Test dataset
│
├── data exploration/
│   ├── eda.ipynb                    # Exploratory data analysis
│   └── smiles_exploration.ipynb     # SMILES structure analysis
│
├── features/
│   ├── __init__.py
│   ├── basic_smiles_feature_generator.py        # Basic molecular features
│   ├── intermediate_smiles_feature_generator.py # Intermediate features
│   └── advanced_smiles_feature_generator.py     # Advanced features
│
├── models (not using SMILES)/
│   ├── linear.ipynb                 # Linear regression model
│   ├── random_forest.ipynb          # Random Forest model
│   ├── xgboost.ipynb                # XGBoost model
│   └── lightgbm.ipynb               # LightGBM model
│
├── models (using SMILES)/
│   ├── xgboost.ipynb                # XGBoost with SMILES features
│   └── lightgbm.ipynb               # LightGBM with SMILES features
│
├── output/
│   └── submission.csv               # Final predictions for submission
│
├── .gitignore
└── README.md
```

---

## 🚀 How to Run the Project

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/melting-point-prediction.git
cd melting-point-prediction
```

### 2️⃣ Install Dependencies
```bash
pip install pandas numpy scikit-learn xgboost lightgbm matplotlib seaborn jupyter
pip install rdkit-pypi  # For SMILES processing
```

*Note: RDKit installation may require conda in some environments:*
```bash
conda install -c conda-forge rdkit
```

### 3️⃣ Explore the Data
Navigate to the `data exploration/` folder and open the notebooks:
- `eda.ipynb` - General exploratory data analysis
- `smiles_exploration.ipynb` - SMILES-specific analysis

### 4️⃣ Generate SMILES Features
The feature generators are modular Python classes located in `features/`:
```python
from features.basic_smiles_feature_generator import BasicSmilesFeatureGenerator
from features.intermediate_smiles_feature_generator import IntermediateSmilesFeatureGenerator
from features.advanced_smiles_feature_generator import AdvancedSmilesFeatureGenerator

# Example usage
generator = BasicSmilesFeatureGenerator()
basic_features = generator.generate(df)
```

### 5️⃣ Train Models
Choose your modeling approach:

**Descriptor-only models:**
- Open notebooks in `models (not using SMILES)/`
- Run: Linear, Random Forest, XGBoost, or LightGBM

**SMILES-enhanced models:**
- Open notebooks in `models (using SMILES)/`
- Run: XGBoost or LightGBM with engineered SMILES features

---

## 📊 Evaluation Metrics

- **RMSE** (Primary Kaggle metric)
- **MAE** (Mean Absolute Error)
- **Cross-validation** for model comparison

**Key Finding:** Tree-based ensemble models (XGBoost, LightGBM) consistently outperformed linear baselines.

---

## 🔍 Key Learnings

- **SMILES feature engineering** can significantly improve performance over descriptor-only approaches
- **Scaling and normalization** are critical for model performance
- **Feature leakage prevention** ensures robust generalization
- **Tree-based models** handle high-dimensional chemical data effectively
- **Modular notebook structure** improves experimentation speed and reproducibility

---

## 🚧 Future Work

- **Graph Neural Networks (GNNs)** using molecular graphs
- **Molecular fingerprints** (Morgan, MACCS keys)
- **Automated feature selection** techniques
- **Model ensembling and stacking** for improved predictions
- **SHAP-based model interpretability** for understanding feature importance
- **Hyperparameter optimization** using Bayesian search or Optuna
- **Deep learning approaches** with molecular representations

---

## 🏆 Kaggle Context

This project was developed as part of a **Kaggle Community competition** focused on predicting chemical properties using machine learning. The competition emphasizes:
- Chemical informatics and cheminformatics techniques
- Regression modeling on molecular data
- Feature engineering from SMILES representations

---

## 📦 Dependencies
```txt
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
xgboost>=1.5.0
lightgbm>=3.3.0
rdkit-pypi>=2022.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
jupyter>=1.0.0
```

---

## 👤 Author

**Aditya Singhal**  
ML/AI Enthusiast

📧 Email: adityasinghal07805@gmail.com  
🔗 LinkedIn: www.linkedin.com/in/aditya-singhal-0b27322ab

Connect with me whenever! I would love to discuss what you have in store further.

---

## 🌟 Acknowledgments

- Kaggle for hosting the competition
- RDKit community for excellent cheminformatics tools
- Open source ML libraries (scikit-learn, XGBoost, LightGBM)

---

**If you find this project useful, feel free to ⭐ the repository!**
