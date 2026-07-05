# IEEE-CIS Fraud Detection: Feature Engineering & LightGBM Modeling

A feature engineering-focused project on the IEEE-CIS Fraud Detection dataset. The primary goal was to identify and engineer high-signal features from raw tabular transaction data to improve fraud detection performance.

## Overview

This project explores whether a machine learning model can effectively identify fraudulent transactions and, more importantly, **which features provide the strongest fraud signals**.

The core work centered on deep exploratory analysis and feature engineering. Many engineered features — particularly binary indicators for missing values — proved highly predictive.

**Final Result:** LightGBM model achieved **0.85056 OOF CV AUC** using Stratified 5-Fold Cross-Validation.

## Repository Structure

| File | Description |
|------|-------------|
| `Fraud Detection - Data Exploration, Feature Engineering, and Feature Selection.ipynb` | Main notebook containing data exploration, fraud signal analysis, feature engineering, and feature selection |
| `Fraud Detection - Training and Testing.ipynb` | Notebook with the final LightGBM model training, 5-fold cross-validation, and evaluation (OOF AUC = 0.85056) |
| `ML_2Detect_FraudulentTransactions_Report` | Written report explaining the data exploration findings, feature engineering decisions, model results, and feature importance analysis |
| `README.md` | This file |

## Dataset

- **Source:** [IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection) (Vesta Corporation)
- Highly imbalanced dataset (~3.5% fraud rate)
- Joined `train_transaction` and `train_identity` on `TransactionID`

## Key Feature Engineering Highlights

A major focus of this project was systematically analyzing columns for fraud signal strength.

Notable work included:
- Creating binary presence/absence features (many missing values carried strong fraud signals)
- Engineering `has_dist2`, `r_email_present`, and multiple `M#` column flags
- Grouping high-cardinality categoricals (`p_email_grouped_v2`, `deviceinfo_grouped_adv`)
- Binning `card3`, `card5`, `id_01`, and `id_11`
- Extracting `hour` from `TransactionDT` (became one of the top features)

## Model & Results

- **Model**: LightGBM with Stratified 5-Fold Cross-Validation + early stopping
- **Metric**: ROC AUC (Out-of-Fold)
- **Performance**: **0.85056 OOF CV AUC**
- Feature importance confirmed that several engineered features ranked among the most predictive

## Key Learnings

- Missing values can be powerful signals in fraud detection (often more informative than the actual values)
- Systematic fraud rate analysis is an effective way to guide feature selection and engineering
- Time-based features (especially hour of day) showed strong predictive power
- LightGBM works very well on mixed tabular data with missing values and imbalance

## Tech Stack

- Python, pandas, numpy
- LightGBM + scikit-learn
- Jupyter Notebooks

## Project Context

This was completed as a school project while working full-time. Emphasis was placed on thoughtful feature engineering and understanding *why* features matter.
