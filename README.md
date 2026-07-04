# IEEE-CIS Fraud Detection: Feature Engineering & LightGBM Modeling

A feature engineering-focused project on the IEEE-CIS Fraud Detection dataset. The primary goal was to identify and engineer high-signal features from raw tabular transaction data to improve fraud detection performance.

## Overview

This project explores whether a machine learning model can effectively identify fraudulent transactions and, more importantly, **which features provide the strongest fraud signals**.

The core work centered on deep exploratory analysis and feature engineering rather than model complexity. Many engineered features — particularly binary indicators for missing values — proved highly predictive.

**Final Result:** LightGBM model achieved **0.85056 OOF CV AUC** using Stratified 5-Fold Cross-Validation.

## Dataset

- **Source:** [IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection) (Vesta Corporation)
- Two main training files joined on `TransactionID`:
  - `train_transaction.csv` (590,540 rows × 394 columns)
  - `train_identity.csv` (144,233 rows × 41 columns)
- Highly imbalanced: ~3.5% fraudulent transactions
- Mix of numerical, categorical, and high-cardinality features

## Key Feature Engineering Work

A major focus of this project was systematically analyzing columns for fraud signal strength and creating new features accordingly.

### Notable Techniques & Discoveries

- **Missing Value Signals**: One of the strongest patterns discovered — the *presence or absence* of a value often carried more signal than the value itself. Created multiple binary "has_X" features.
- **Distance Features** (`dist1`, `dist2`): Converted to binary presence flags (`has_dist1`, `has_dist2`).
- **M1–M9 Columns**: Created binary flags; kept high-signal engineered versions (e.g., `m4_checked`, `m6_checked`).
- **Email Domains**: Grouped `P_emaildomain` into meaningful categories (`p_email_grouped_v2`) and created `r_email_present`.
- **Card Features**: Binned `card3` and `card5`; kept `card4` (vendor) and `card6` (type) as categories.
- **Transaction Timing** (`TransactionDT`): Engineered `hour` feature, which became one of the most important predictors.
- **Device Information**: Grouped high-cardinality `DeviceInfo` into `deviceinfo_grouped_adv`.
- **Address & ID Fields**: Created presence flags and binned where useful (`id_01`, `id_11`).

Many of these engineered features ranked among the **top features** by importance in the final model.

## Model & Evaluation

- **Algorithm**: LightGBM (chosen for native categorical support and strong performance on imbalanced tabular data)
- **Validation**: Stratified 5-Fold Cross-Validation with early stopping
- **Metric**: ROC AUC (Out-of-Fold)
- **Result**: **0.85056 OOF CV AUC**

Feature importance analysis confirmed that both original and engineered features contributed meaningfully to predictions.

## Key Learnings

- Missing values are not just noise — in this dataset they often represented strong behavioral signals.
- Systematic fraud signal analysis (comparing fraud rates when values are present vs. missing) is an effective feature selection strategy.
- Time-based features (especially hour of day) can be highly predictive in fraud detection.
- LightGBM handles mixed data types and missing values gracefully, making it well-suited for real-world tabular fraud problems.

## Tech Stack

- **Language**: Python
- **Core Libraries**: pandas, numpy, LightGBM, scikit-learn
- **Analysis**: Custom feature engineering pipeline with fraud rate analysis

## Project Context

This was completed as a school project while working full-time. The emphasis was placed on thoughtful feature engineering and understanding *why* certain features matter, rather than chasing leaderboard scores.

---

**Note**: This repository focuses on the feature engineering methodology and analysis process. The final modeling code follows a standard LightGBM + Stratified K-Fold setup.
