# Auto-Insurance-Claim-Prediction
**EE559 Course Project — University of Southern California, Spring 2026**

## Overview
This project applies supervised and unsupervised machine learning to predict 
whether a policyholder will file an auto insurance claim based on demographic 
and driving history features. Three classifiers are compared, hyperparameter 
tuned, and evaluated. K-Means clustering is applied to discover natural risk 
segments within the policyholder population.

## Repository Structure
```
Auto-Insurance-Claim-Prediction/
├── Main.ipynb       # Main notebook
├── customer-data.csv        # Dataset
├── README.md                
└── Project_Report.pdf # Final report
``` 

## Pipeline
1. Exploratory Data Analysis (EDA)
2. Preprocessing & Feature Engineering
3. Classification — Logistic Regression, Random Forest, MLP
4. Hyperparameter Tuning — RandomizedSearchCV (5-fold CV)
5. Clustering — K-Means (k=2, selected via elbow + silhouette)
6. Cluster Validation & PCA Visualization

## Dataset
- **Source:** [Customer Data — Kaggle](https://www.kaggle.com/datasets/racholsan/customer-data)
- **Size:** 10,000 records, 15 features (after preprocessing)
- **Target:** `outcome` — binary (0 = no claim, 1 = claim)
- **Class Distribution:** 68.7% no claim, 31.3% claim

## Results

### Classification (Tuned Models)
| Model | Accuracy | F1 | ROC-AUC |
|---|---|---|---|
| Logistic Regression | 0.798 | 0.723 | **0.886** |
| Random Forest | 0.818 | **0.734** | 0.875 |
| MLP | **0.828** | 0.718 | 0.883 |

### Clustering (K-Means, k=2)
| Cluster | Size | Claim Rate |
|---|---|---|
| Cluster 0 (High Risk) | 3,608 | 55.8% |
| Cluster 1 (Low Risk) | 4,392 | 11.2% |

## Key Findings
- Logistic Regression achieved the highest ROC-AUC (0.886) and claim recall (0.84) — most practically useful for insurance pricing
- `driving_experience` was the single most important predictive feature (importance = 0.234)
- K-Means discovered a 5x difference in claim rate between risk segments purely from demographic features — without accessing outcome labels
- Adding cluster membership as a feature produced negligible improvement, confirming the classifier already captures the underlying risk structure

## Requirements
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## How to Run
1. Clone the repository
```bash
git clone https://github.com/jwong5-code/Auto-Insurance-Claim-Prediction
```
2. Open and run the notebook top to bottom
```bash
jupyter notebook EE559Project.ipynb
```
The dataset (`customer-data.csv`) is included — no additional downloads required.

## Academic Integrity
All modeling and analysis code is original. Standard libraries (scikit-learn, pandas, matplotlib, seaborn) were used throughout.
