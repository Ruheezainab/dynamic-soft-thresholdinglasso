# Dynamic Soft-Thresholding for Feature Selection in High-Dimensional Regression

# Overview

This project implements a Dynamic LASSO optimizer using Proximal Gradient Descent with soft-thresholding to perform feature selection in high-dimensional regression.

The objective is to reduce the number of features while maintaining predictive accuracy compared to baseline models.

The project compares the following models:

Linear Regression
Ridge Regression
LASSO Regression
Dynamic LASSO (custom optimizer)
Dataset used: House Prices – Advanced Regression Techniques (Kaggle).

 # Mathematical Formulation
The optimization objective is:

min (1/2n)||y − Xw||² + λ||w||₁

Where:

X = Feature matrix
y = Target variable (SalePrice)
w = Model coefficients
λ = Regularization parameter
n = Number of samples

Feature selection occurs through the soft-thresholding proximal operator:

SoftThreshold(x, λ) = sign(x) × max(|x| − λ, 0)

Small coefficients shrink to zero, producing sparse models.

# Dataset
Dataset: House Prices – Advanced Regression Techniques

Download from Kaggle:

https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data

After downloading, place:

train.csv

inside the data/ folder.

Dataset details:

Samples: 1460
Features: 80
Numerical features: 37
Categorical features: 43
Target variable: SalePrice

# Project Structure
Dynamic-Soft-Thresholding-LASSO/ │ ├── src/ │ ├── main.py │ ├── preprocessing.py │ ├── dynamic_lasso.py │ ├── baseline_models.py │ ├── evaluation.py │ └── visualization.py │ ├── data/ │ └── train.csv │ ├── results/ │ ├── model_comparison.png │ ├── convergence.png │ ├── lambda_decay.png │ ├── sparsity_curve.png │ ├── actual_vs_predicted.png │ ├── residuals.png │ ├── model_comparison.csv │ ├── coefficients.csv │ └── training_history.csv │ ├── requirements.txt └── README.md

# Installation
Install the required Python libraries:

pip install -r requirements.txt

requirements.txt
numpy
pandas
scikit-learn
matplotlib
seaborn

# Running the Project
Run the full pipeline using:

python src/main.py

The script performs:

Data loading and preprocessing
Training baseline models
Training Dynamic LASSO optimizer
Model evaluation
Accuracy constraint checking
Visualization generation
Saving results

# Data Preprocessing
The preprocessing pipeline includes:

Median imputation for numerical features
Filling missing categorical values with "Missing"
Ordinal encoding of categorical variables
Standardization using StandardScaler
Train-test split (80% training / 20% testing)

# Models Evaluated
Linear Regression – Baseline model
Ridge Regression – L2 regularization
LASSO Regression – L1 regularization
Dynamic LASSO – Custom proximal gradient optimizer

Evaluation metrics used:

RMSE
MAE
R² score
Feature sparsity

# Example Results
Typical results from the experiment:

Model | Test RMSE | R² | Sparsity Linear Regression | ~34691 | 0.843 | 0% Ridge | ~34562 | 0.844 | 0% LASSO | ~34690 | 0.843 | ~1% Dynamic LASSO | ~37286 | 0.818 | 60%

Feature selection result:

Total features: 80
Selected features: 32
Eliminated features: 48
Sparsity: 60%

Accuracy constraint:

Baseline RMSE: 34647
Dynamic LASSO RMSE: 37286
Deviation: 7.61%
Tolerance: 15%

Result: Constraint satisfied.

# Generated Outputs
The following outputs are automatically created inside the results/ folder.

Visualization Files:

model_comparison.png
convergence.png
lambda_decay.png
sparsity_curve.png
actual_vs_predicted.png
residuals.png

Data Files:

model_comparison.csv
coefficients.csv
training_history.csv

# Conclusion
The Dynamic LASSO optimizer demonstrates how proximal gradient methods can be applied for feature selection in high-dimensional regression problems.

Key outcomes:

Significant feature reduction
Acceptable predictive performance
Demonstration of numerical optimization techniques in machine learning
# Author
Developed for coursework on Numerical Optimization and Machine Learning.

Python Version: 3.8+
