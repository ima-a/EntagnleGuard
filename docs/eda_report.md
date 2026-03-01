# Credit Card Fraud EDA Report

## Source Script
This report summarizes the exploratory analysis implemented in `credit_card_eda.py`.

## Dataset Context
- Dataset: Credit card fraud transactions (Kaggle ULB dataset expected as `creditcard.csv`)
- Records: ~284,807 transactions
- Target column: `Class` (`0` = legitimate, `1` = fraud)
- Features:
  - `V1`–`V28`: PCA-transformed/anonymized features
  - `Time`: seconds from first transaction
  - `Amount`: transaction value

## EDA Workflow Covered in the Script

### 1) Data Loading and Initial Checks
- Tries to read `creditcard.csv`.
- If missing, creates synthetic demo data so the EDA pipeline can still run.
- Prints:
  - data shape
  - dtype distribution
  - missing value count
  - duplicate row count
  - descriptive statistics for `Time`, `Amount`, and `Class`

### 2) Class Imbalance Analysis
- Computes fraud vs legitimate counts and fraud percentage.
- Highlights severe class imbalance.
- Saves `01_class_distribution.png` with:
  - bar chart (log scale)
  - pie chart of fraud proportion

### 3) Transaction Amount Analysis
- Compares `Amount` distribution for fraud vs legitimate classes.
- Produces:
  - histogram (log x-scale)
  - boxplot (log y-scale)
  - violin plot (`log1p`)
  - CDF comparison
- Saves `02_amount_analysis.png`.

### 4) Time-Based Analysis
- Creates `Time_Hours = Time / 3600`.
- Analyzes temporal transaction density and fraud-rate variation across time bins.
- Saves `03_time_analysis.png`.

### 5) PCA Feature Distribution Analysis (`V1`–`V28`)
- Computes feature-class separation using Cohen’s d for each PCA feature.
- Lists top discriminating features.
- Plots KDE distributions for all PCA features by class.
- Saves `04_pca_feature_distributions.png`.

### 6) Correlation Analysis
- Computes Pearson correlation of all features with `Class`.
- Builds:
  - horizontal bar chart of feature correlations to `Class`
  - heatmap for top class-correlated features + `Time` + `Amount`
- Saves `05_correlation_analysis.png`.

### 7) Outlier Analysis
- Uses IQR method with conservative threshold (`Q1 - 3*IQR`, `Q3 + 3*IQR`).
- Computes fraud rate among outliers per feature.
- Reports top features where outliers are most fraud-heavy.
- Saves top-feature class-wise boxplots in `06_boxplots_top_features.png`.

### 8) Pairwise Feature Interaction View
- Uses top 4 most discriminating features.
- Builds pair-plot style matrix:
  - KDE on diagonal
  - scatter plots off-diagonal
- Saves `07_pair_plot_top_features.png`.

### 9) Final Summary Dashboard
- Combines key metrics and visuals into one dashboard:
  - class imbalance
  - top Cohen’s d features
  - amount distribution
  - fraud rate over time
  - key stats table
- Saves `08_summary_dashboard.png`.

## Main Insights Captured by the Script
- Fraud detection is highly imbalanced; accuracy alone is misleading.
- Amount distributions are right-skewed; `log1p(Amount)` is suggested.
- Fraud rate can vary by time window; hour/time features may help.
- Certain PCA features show strong separation (high Cohen’s d).
- Outlier behavior in some features is more fraud-concentrated.

## Modeling-Oriented Recommendations (from script output)
- Use imbalance-aware strategy (class weights / resampling).
- Evaluate with Precision-Recall, F1, and AUC-PR.
- Engineer features like hour-of-day and transformed amount.
- Start with robust baseline models (e.g., tree-based models) and threshold tuning.

## Output Note
The script currently writes figures to `./eda_outputs/` by default.
In this workspace, generated images have been organized under `datset/`.
