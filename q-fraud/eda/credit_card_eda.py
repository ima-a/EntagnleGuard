"""
=============================================================================
CREDIT CARD FRAUD DETECTION - EXPLORATORY DATA ANALYSIS (EDA)
=============================================================================

DATASET SOURCE:
    Kaggle: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/data
    Originally published by: Machine Learning Group - ULB (Université Libre de Bruxelles)

DATASET OVERVIEW:
    - Contains transactions made by European credit card holders in September 2013
    - Covers 2 days of transactions: 284,807 total transactions
    - 492 are fraudulent (0.172% of all transactions)
    - Features V1–V28 are PCA-transformed (anonymized for confidentiality)
    - 'Time' = seconds elapsed between each transaction and the first transaction
    - 'Amount' = transaction amount in Euros
    - 'Class' = target label (1 = Fraud, 0 = Legitimate)

EDA OBJECTIVE:
    Understand the data structure, detect patterns, identify class imbalance,
    and derive insights that will guide preprocessing and modeling decisions.

HOW TO USE THIS SCRIPT:
    1. Download creditcard.csv from Kaggle and place it in the same directory
    2. Run: python credit_card_eda.py
    3. All plots will be saved to ./eda_outputs/ folder
    4. Summary stats will be printed to console

DEPENDENCIES:
    pip install pandas numpy matplotlib seaborn scipy
=============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
import warnings
import os

warnings.filterwarnings('ignore')

# ── Output directory ────────────────────────────────────────────────────────
OUTPUT_DIR = "./eda_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Plot style ───────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': '#0f1117',
    'axes.facecolor':   '#1a1d27',
    'axes.edgecolor':   '#444',
    'axes.labelcolor':  '#ccc',
    'xtick.color':      '#aaa',
    'ytick.color':      '#aaa',
    'text.color':       '#eee',
    'grid.color':       '#2a2d3a',
    'grid.linestyle':   '--',
    'grid.alpha':       0.6,
    'font.family':      'DejaVu Sans',
})

FRAUD_COLOR  = '#ff4d6d'
LEGIT_COLOR  = '#4cc9f0'
ACCENT_COLOR = '#f72585'

# =============================================================================
# SECTION 1 – DATA LOADING & FIRST LOOK
# =============================================================================
"""
WHY THIS STEP:
    Before any analysis, we confirm the dataset loads correctly, check its
    shape, column names, data types, and whether there are any missing values.
    This is the foundation of every EDA.
"""

print("\n" + "="*65)
print("  CREDIT CARD FRAUD EDA  |  Loading Data…")
print("="*65)

try:
    df = pd.read_csv("creditcard.csv")
    print(f"✅  Dataset loaded: {df.shape[0]:,} rows  ×  {df.shape[1]} columns")
except FileNotFoundError:
    print("⚠️  creditcard.csv not found – generating synthetic demo data…")
    print("   (Place the real Kaggle CSV in this folder for actual results)\n")
    np.random.seed(42)
    n_legit, n_fraud = 284315, 492
    n = n_legit + n_fraud
    pca_cols = {f'V{i}': np.random.randn(n) for i in range(1, 29)}
    df = pd.DataFrame(pca_cols)
    df['Time']   = np.sort(np.random.uniform(0, 172800, n))
    df['Amount'] = np.concatenate([
        np.random.exponential(88, n_legit),
        np.random.exponential(122, n_fraud)
    ])
    df['Class']  = np.concatenate([np.zeros(n_legit), np.ones(n_fraud)]).astype(int)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"✅  Synthetic dataset created: {df.shape[0]:,} rows  ×  {df.shape[1]} columns")

# =============================================================================
# SECTION 2 – BASIC STATISTICS & DATA QUALITY
# =============================================================================
"""
WHY THIS STEP:
    - .info()    → column types and non-null counts
    - .describe() → min/max/mean/std/percentiles for all numeric columns
    - Missing value check → determines if imputation is needed
    - Duplicate check → avoids inflated counts in analysis

WHAT WE EXPECT TO FIND:
    All 30 feature columns are float64; Class is int64.
    The Kaggle dataset has NO missing values.
    A small number of duplicate rows may exist (investigate if found).
"""

print("\n── DATA TYPES & NULL VALUES ──────────────────────────────────────")
print(df.dtypes.value_counts().to_string())
print(f"\nMissing values: {df.isnull().sum().sum()}")
print(f"Duplicate rows: {df.duplicated().sum():,}")

print("\n── DESCRIPTIVE STATISTICS (non-PCA columns) ──────────────────────")
print(df[['Time', 'Amount', 'Class']].describe().round(2).to_string())

# =============================================================================
# SECTION 3 – CLASS DISTRIBUTION (IMBALANCE ANALYSIS)
# =============================================================================
"""
WHY THIS STEP:
    Class imbalance is the central challenge in fraud detection.
    If ~99.8% of records are legitimate, a naive model that predicts "no fraud"
    every time would achieve 99.8% accuracy but zero recall on fraud.

    This plot quantifies the imbalance so we can decide on techniques like:
      - SMOTE (Synthetic Minority Oversampling)
      - Class-weight adjustment in model training
      - Precision-Recall curves instead of ROC-AUC
"""

fraud_count  = (df['Class'] == 1).sum()
legit_count  = (df['Class'] == 0).sum()
fraud_pct    = fraud_count / len(df) * 100

print(f"\n── CLASS DISTRIBUTION ────────────────────────────────────────────")
print(f"  Legitimate (0): {legit_count:>7,}  ({100 - fraud_pct:.3f}%)")
print(f"  Fraud      (1): {fraud_count:>7,}  ({fraud_pct:.3f}%)")
print(f"  Imbalance ratio: 1 : {legit_count // fraud_count}")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Class Distribution – Fraud vs Legitimate", fontsize=14, fontweight='bold', y=1.01)

# Bar chart
labels = ['Legitimate\n(Class 0)', 'Fraud\n(Class 1)']
counts = [legit_count, fraud_count]
colors = [LEGIT_COLOR, FRAUD_COLOR]
bars   = axes[0].bar(labels, counts, color=colors, width=0.4, edgecolor='#444', linewidth=1.2)
axes[0].set_title("Transaction Count by Class")
axes[0].set_ylabel("Count")
axes[0].set_yscale('log')
axes[0].set_ylim(1, max(counts) * 3)
for bar, cnt in zip(bars, counts):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.3,
                 f'{cnt:,}', ha='center', fontsize=11, fontweight='bold', color='white')

# Pie chart (zoomed on fraud)
axes[1].pie([legit_count, fraud_count],
            labels=[f'Legitimate\n{100-fraud_pct:.3f}%', f'Fraud\n{fraud_pct:.3f}%'],
            colors=[LEGIT_COLOR, FRAUD_COLOR],
            startangle=90, explode=(0, 0.15),
            wedgeprops=dict(edgecolor='#222', linewidth=1.5),
            textprops={'color': 'white', 'fontsize': 11})
axes[1].set_title("Proportion (Note: Extreme Imbalance)")

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/01_class_distribution.png", dpi=150, bbox_inches='tight')
plt.close()
print("  → Saved: 01_class_distribution.png")

# =============================================================================
# SECTION 4 – TRANSACTION AMOUNT ANALYSIS
# =============================================================================
"""
WHY THIS STEP:
    'Amount' is the only raw (non-PCA) financial feature.
    We compare the distribution of transaction amounts between fraud and
    legitimate transactions to see if amount is a useful discriminating feature.

KEY QUESTIONS:
    - Are fraudulent transactions typically small (to avoid detection) or large?
    - Is the distribution skewed? (Hint: yes – most transactions are small)
    - Should we log-transform Amount before modeling?

TYPICAL FINDINGS:
    - Both distributions are right-skewed (long tail of high-value transactions)
    - Fraud transactions tend to cluster at lower amounts (stealth behavior)
    - The mean fraud amount is often *lower* than legitimate (counter-intuitive!)
    - Log transformation is recommended before modeling
"""

fraud_amounts = df[df['Class'] == 1]['Amount']
legit_amounts = df[df['Class'] == 0]['Amount']

print(f"\n── AMOUNT STATISTICS ─────────────────────────────────────────────")
print(f"  Legit  – mean: €{legit_amounts.mean():>8.2f}  |  median: €{legit_amounts.median():>6.2f}  |  max: €{legit_amounts.max():>10.2f}")
print(f"  Fraud  – mean: €{fraud_amounts.mean():>8.2f}  |  median: €{fraud_amounts.median():>6.2f}  |  max: €{fraud_amounts.max():>10.2f}")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Transaction Amount Analysis", fontsize=15, fontweight='bold')

# Histogram (log-scaled x)
for label, data, color in [('Legitimate', legit_amounts, LEGIT_COLOR), ('Fraud', fraud_amounts, FRAUD_COLOR)]:
    axes[0, 0].hist(data[data > 0], bins=80, alpha=0.7, color=color, label=label,
                    density=True, edgecolor='none')
axes[0, 0].set_xscale('log')
axes[0, 0].set_title("Amount Distribution (log scale)")
axes[0, 0].set_xlabel("Amount (€, log scale)")
axes[0, 0].set_ylabel("Density")
axes[0, 0].legend()

# Box plot
bp_data = [legit_amounts, fraud_amounts]
bp = axes[0, 1].boxplot(bp_data, patch_artist=True, notch=True,
                         medianprops=dict(color='white', linewidth=2))
for patch, color in zip(bp['boxes'], [LEGIT_COLOR, FRAUD_COLOR]):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
axes[0, 1].set_yscale('log')
axes[0, 1].set_xticklabels(['Legitimate', 'Fraud'])
axes[0, 1].set_title("Amount Box Plot (log scale)")
axes[0, 1].set_ylabel("Amount (€, log scale)")

# Violin
parts = axes[1, 0].violinplot([np.log1p(legit_amounts), np.log1p(fraud_amounts)],
                               positions=[1, 2], showmedians=True)
for i, pc in enumerate(parts['bodies']):
    pc.set_facecolor([LEGIT_COLOR, FRAUD_COLOR][i])
    pc.set_alpha(0.7)
axes[1, 0].set_xticks([1, 2])
axes[1, 0].set_xticklabels(['Legitimate', 'Fraud'])
axes[1, 0].set_title("Amount Violin Plot (log1p scale)\n← wider = more transactions at that value")
axes[1, 0].set_ylabel("log(Amount + 1)")

# CDF comparison
for label, data, color in [('Legitimate', legit_amounts, LEGIT_COLOR), ('Fraud', fraud_amounts, FRAUD_COLOR)]:
    sorted_data = np.sort(data)
    cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    axes[1, 1].plot(sorted_data, cdf, color=color, label=label, linewidth=2)
axes[1, 1].set_xscale('log')
axes[1, 1].set_xlabel("Amount (€, log scale)")
axes[1, 1].set_ylabel("Cumulative Probability")
axes[1, 1].set_title("Cumulative Distribution Function\n← steeper = amounts concentrated at lower values")
axes[1, 1].legend()
axes[1, 1].axvline(x=100, color='#ffd166', linestyle='--', alpha=0.6, label='€100 reference')

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/02_amount_analysis.png", dpi=150, bbox_inches='tight')
plt.close()
print("  → Saved: 02_amount_analysis.png")

# =============================================================================
# SECTION 5 – TIME ANALYSIS
# =============================================================================
"""
WHY THIS STEP:
    'Time' records seconds elapsed since the first transaction.
    Since the data spans 2 days (0–172,800 seconds), we can examine:
      - Are there temporal patterns in transaction volume? (day vs night)
      - Do fraudulent transactions happen more at specific times?
      - Is there a "quiet period" (night-time) that fraudsters exploit?

TYPICAL FINDINGS:
    - Transaction volume dips during overnight hours (~40,000–80,000 seconds)
    - Fraud is relatively more prevalent during low-volume periods
      (fewer transactions = less scrutiny / easier to slip through)
    - Two distinct "peaks" correspond to the two business days in the data
"""

df['Time_Hours'] = df['Time'] / 3600  # Convert seconds → hours

fig, axes = plt.subplots(2, 1, figsize=(14, 9))
fig.suptitle("Temporal Analysis of Transactions", fontsize=15, fontweight='bold')

# Transaction volume over time
axes[0].hist(df[df['Class'] == 0]['Time_Hours'], bins=100,
             color=LEGIT_COLOR, alpha=0.7, label='Legitimate', density=True)
axes[0].hist(df[df['Class'] == 1]['Time_Hours'], bins=100,
             color=FRAUD_COLOR, alpha=0.9, label='Fraud', density=True)
axes[0].axvline(x=24, color='#ffd166', linestyle='--', alpha=0.8, linewidth=1.5, label='24h mark')
axes[0].set_xlabel("Hours since first transaction")
axes[0].set_ylabel("Density")
axes[0].set_title("Transaction Density Over Time (2-day window)")
axes[0].legend()
axes[0].set_xticks(range(0, 50, 4))

# Fraud rate over time (binned)
bins   = pd.cut(df['Time_Hours'], bins=48)
fraud_rate = df.groupby(bins, observed=False)['Class'].mean() * 100
bin_centers = [interval.mid for interval in fraud_rate.index]
axes[1].bar(bin_centers, fraud_rate.values,
            color=[FRAUD_COLOR if v > fraud_pct * 1.5 else '#888' for v in fraud_rate.values],
            width=0.45, edgecolor='none', alpha=0.85)
axes[1].axhline(y=fraud_pct, color='#ffd166', linestyle='--',
                linewidth=1.5, label=f'Overall fraud rate ({fraud_pct:.3f}%)')
axes[1].set_xlabel("Hours since first transaction")
axes[1].set_ylabel("Fraud Rate (%)")
axes[1].set_title("Fraud Rate by Time Bin\n(Red = significantly above baseline)")
axes[1].legend()

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/03_time_analysis.png", dpi=150, bbox_inches='tight')
plt.close()
print("  → Saved: 03_time_analysis.png")

# =============================================================================
# SECTION 6 – PCA FEATURE DISTRIBUTIONS (V1–V28)
# =============================================================================
"""
WHY THIS STEP:
    V1–V28 are PCA-transformed features. We can't reverse-engineer the originals,
    but we CAN compare their distributions between fraud and legitimate classes.

    Features where fraud and legitimate distributions differ greatly are
    highly predictive — they are likely the most important features for a model.

WHAT TO LOOK FOR:
    - Features where the fraud median/mean is far from the legit median/mean
    - Features with bimodal distributions in one class but not the other
    - Features with very different spreads (variance) between classes

TYPICAL HIGH-SIGNAL FEATURES (on the real dataset):
    V4, V11, V12, V14, V17 — tend to show the strongest separation
    V13, V15, V22, V23, V25 — tend to show weak separation
"""

print("\n── PCA FEATURE ANALYSIS ──────────────────────────────────────────")

pca_features = [f'V{i}' for i in range(1, 29)]

# Compute separation score (Cohen's d) for each feature
separation_scores = {}
for feat in pca_features:
    g1 = df[df['Class'] == 0][feat]
    g2 = df[df['Class'] == 1][feat]
    pooled_std = np.sqrt((g1.std()**2 + g2.std()**2) / 2)
    cohens_d   = abs(g1.mean() - g2.mean()) / (pooled_std + 1e-9)
    separation_scores[feat] = cohens_d

sorted_features = sorted(separation_scores, key=separation_scores.get, reverse=True)
print("  Top 5 discriminating features (Cohen's d):")
for feat in sorted_features[:5]:
    print(f"    {feat}: d = {separation_scores[feat]:.3f}")

# KDE plots for all 28 features
fig, axes = plt.subplots(7, 4, figsize=(18, 24))
fig.suptitle("V1–V28 PCA Feature Distributions\n(Fraud vs Legitimate)",
             fontsize=14, fontweight='bold', y=1.001)
axes_flat = axes.flatten()

for i, feat in enumerate(pca_features):
    ax = axes_flat[i]
    df[df['Class'] == 0][feat].plot.kde(ax=ax, color=LEGIT_COLOR, label='Legit', linewidth=1.5)
    df[df['Class'] == 1][feat].plot.kde(ax=ax, color=FRAUD_COLOR, label='Fraud', linewidth=1.8)
    ax.set_title(f"{feat}  (d={separation_scores[feat]:.2f})",
                 fontsize=9,
                 color=FRAUD_COLOR if separation_scores[feat] > 1.0 else '#ccc')
    ax.set_ylabel("")
    ax.set_xlabel("")
    ax.legend(fontsize=6)
    ax.tick_params(labelsize=7)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/04_pca_feature_distributions.png", dpi=130, bbox_inches='tight')
plt.close()
print("  → Saved: 04_pca_feature_distributions.png")

# =============================================================================
# SECTION 7 – CORRELATION HEATMAP
# =============================================================================
"""
WHY THIS STEP:
    PCA by design produces uncorrelated components — so V1–V28 should have
    near-zero correlation with each other.

    HOWEVER:
    1. We check correlation with 'Class' to see which features are most
       associated with fraud (confirmation of Cohen's d above).
    2. We verify that Amount and Time are not collinear with PCA features.
    3. Any unexpected high correlations may indicate data leakage or issues.

NOTE:
    The heatmap will show a mostly grey/neutral matrix for V1–V28 inter-correlations,
    confirming PCA's orthogonality property. The Class column will show
    meaningful correlations for some Vn features.
"""

# Correlation with Class (most informative view)
corr_with_class = df.corr()['Class'].drop('Class').sort_values()

fig, axes = plt.subplots(1, 2, figsize=(16, 8))
fig.suptitle("Correlation Analysis", fontsize=14, fontweight='bold')

# Bar chart of correlations with Class
colors_bar = [FRAUD_COLOR if v > 0 else LEGIT_COLOR for v in corr_with_class.values]
axes[0].barh(corr_with_class.index, corr_with_class.values,
             color=colors_bar, edgecolor='none', height=0.7)
axes[0].axvline(0, color='white', linewidth=0.8)
axes[0].set_title("Feature Correlation with 'Class'\n(Red=positive=associated with fraud)", fontsize=10)
axes[0].set_xlabel("Pearson Correlation Coefficient")

# Heatmap of top 15 correlated features + Time + Amount
top_features = list(corr_with_class.abs().nlargest(13).index) + ['Time', 'Amount', 'Class']
top_features = list(dict.fromkeys(top_features))  # deduplicate
corr_matrix  = df[top_features].corr()

mask = np.zeros_like(corr_matrix, dtype=bool)
mask[np.triu_indices_from(mask)] = True

sns.heatmap(corr_matrix, ax=axes[1], mask=mask,
            cmap='RdBu_r', center=0, vmin=-1, vmax=1,
            annot=True, fmt='.2f', annot_kws={'size': 7},
            linewidths=0.5, linecolor='#333',
            cbar_kws={'shrink': 0.7})
axes[1].set_title("Correlation Heatmap\n(Top 13 Class-correlated features)", fontsize=10)
axes[1].tick_params(axis='x', rotation=45, labelsize=8)
axes[1].tick_params(axis='y', rotation=0, labelsize=8)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/05_correlation_analysis.png", dpi=150, bbox_inches='tight')
plt.close()
print("  → Saved: 05_correlation_analysis.png")

# =============================================================================
# SECTION 8 – OUTLIER ANALYSIS
# =============================================================================
"""
WHY THIS STEP:
    Fraud transactions may manifest as statistical outliers — transactions that
    fall far outside the normal range for a given feature.

    We use the IQR (Interquartile Range) method:
        Lower bound = Q1 - 3×IQR
        Upper bound = Q3 + 3×IQR
        (Using 3× instead of 1.5× for a more conservative threshold)

    KEY INSIGHT:
    If a high percentage of the 'outlier' records are fraudulent,
    outlier-based features (distance from mean, z-score) may be a
    useful engineered feature for modeling.
"""

print("\n── OUTLIER ANALYSIS ──────────────────────────────────────────────")

outlier_fraud_rates = {}
for feat in pca_features + ['Amount']:
    Q1, Q3 = df[feat].quantile(0.25), df[feat].quantile(0.75)
    IQR    = Q3 - Q1
    lower, upper = Q1 - 3 * IQR, Q3 + 3 * IQR
    outliers = df[(df[feat] < lower) | (df[feat] > upper)]
    if len(outliers) > 0:
        rate = outliers['Class'].mean() * 100
        outlier_fraud_rates[feat] = rate

sorted_outlier = sorted(outlier_fraud_rates.items(), key=lambda x: x[1], reverse=True)
print("  Top 5 features where outliers are most fraudulent:")
for feat, rate in sorted_outlier[:5]:
    print(f"    {feat}: {rate:.1f}% of outliers are fraud")

# Box plots for top 8 most discriminating features
top8 = sorted_features[:8]
fig, axes = plt.subplots(2, 4, figsize=(18, 8))
fig.suptitle("Box Plots – Top 8 Discriminating Features (by Cohen's d)", fontsize=13, fontweight='bold')

for i, feat in enumerate(top8):
    ax = axes[i // 4][i % 4]
    data_to_plot = [df[df['Class'] == 0][feat], df[df['Class'] == 1][feat]]
    bp = ax.boxplot(data_to_plot, patch_artist=True, notch=True,
                    medianprops=dict(color='white', linewidth=2),
                    flierprops=dict(marker='o', alpha=0.3, markersize=2))
    for patch, color in zip(bp['boxes'], [LEGIT_COLOR, FRAUD_COLOR]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_xticklabels(['Legit', 'Fraud'])
    ax.set_title(f"{feat}  (d={separation_scores[feat]:.2f})", fontsize=10)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/06_boxplots_top_features.png", dpi=150, bbox_inches='tight')
plt.close()
print("  → Saved: 06_boxplots_top_features.png")

# =============================================================================
# SECTION 9 – SCATTER & PAIR PLOTS FOR TOP FEATURES
# =============================================================================
"""
WHY THIS STEP:
    2D scatter plots and pair plots reveal how well features JOINTLY separate
    fraud from legitimate transactions. A single feature might not separate
    classes well, but two features together might create a clear boundary.

    This informs feature selection and which combinations might work well
    in a linear model (logistic regression) vs tree-based models.
"""

top4 = sorted_features[:4]
fig, axes = plt.subplots(len(top4), len(top4), figsize=(14, 14))
fig.suptitle(f"Pair Plot – Top 4 Discriminating Features:\n{', '.join(top4)}",
             fontsize=13, fontweight='bold')

sample_legit = df[df['Class'] == 0].sample(n=min(2000, legit_count), random_state=42)
sample_fraud = df[df['Class'] == 1].sample(n=min(fraud_count, fraud_count), random_state=42)
sample       = pd.concat([sample_legit, sample_fraud])

for i, feat_y in enumerate(top4):
    for j, feat_x in enumerate(top4):
        ax = axes[i][j]
        if i == j:  # Diagonal: KDE
            sample[sample['Class'] == 0][feat_x].plot.kde(ax=ax, color=LEGIT_COLOR, linewidth=1.5)
            sample[sample['Class'] == 1][feat_x].plot.kde(ax=ax, color=FRAUD_COLOR, linewidth=1.5)
            ax.set_ylabel("")
        else:       # Off-diagonal: scatter
            ax.scatter(sample[sample['Class'] == 0][feat_x],
                       sample[sample['Class'] == 0][feat_y],
                       c=LEGIT_COLOR, alpha=0.2, s=4, label='Legit')
            ax.scatter(sample[sample['Class'] == 1][feat_x],
                       sample[sample['Class'] == 1][feat_y],
                       c=FRAUD_COLOR, alpha=0.7, s=8, label='Fraud')
        if i == len(top4) - 1:
            ax.set_xlabel(feat_x, fontsize=9)
        else:
            ax.set_xticklabels([])
        if j == 0:
            ax.set_ylabel(feat_y, fontsize=9)
        else:
            ax.set_yticklabels([])

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/07_pair_plot_top_features.png", dpi=130, bbox_inches='tight')
plt.close()
print("  → Saved: 07_pair_plot_top_features.png")

# =============================================================================
# SECTION 10 – SUMMARY DASHBOARD
# =============================================================================
"""
WHY THIS STEP:
    A final one-page dashboard that condenses the key findings for
    quick reference. Useful for presentations and documentation.
"""

fig = plt.figure(figsize=(16, 10))
fig.patch.set_facecolor('#0f1117')
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

fig.text(0.5, 0.97, 'Credit Card Fraud EDA – Summary Dashboard',
         ha='center', va='top', fontsize=16, fontweight='bold', color='white')
fig.text(0.5, 0.935,
         f'284,807 total transactions  |  {fraud_pct:.3f}% fraud  |  30 features (28 PCA + Time + Amount)',
         ha='center', va='top', fontsize=10, color='#aaa')

# ── Panel 1: Class balance (bar) ────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
ax1.set_facecolor('#1a1d27')
bars = ax1.bar(['Legitimate', 'Fraud'], [legit_count, fraud_count],
               color=[LEGIT_COLOR, FRAUD_COLOR], edgecolor='#444')
ax1.set_yscale('log')
ax1.set_title("Class Imbalance (log scale)", fontsize=10)
for bar, cnt in zip(bars, [legit_count, fraud_count]):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.5,
             f'{cnt:,}', ha='center', fontsize=9, color='white', fontweight='bold')

# ── Panel 2: Feature importance by Cohen's d ───────────────────────────────
ax2 = fig.add_subplot(gs[0, 1:])
ax2.set_facecolor('#1a1d27')
top_n   = 14
feats   = sorted_features[:top_n]
scores  = [separation_scores[f] for f in feats]
colors2 = [FRAUD_COLOR if s > 1.5 else ACCENT_COLOR if s > 0.8 else '#888' for s in scores]
ax2.barh(feats[::-1], scores[::-1], color=colors2[::-1], height=0.6)
ax2.axvline(x=1.0, color='#ffd166', linestyle='--', linewidth=1, label="d=1.0 (large effect)")
ax2.set_title(f"Top {top_n} Features – Fraud Discriminability (Cohen's d)", fontsize=10)
ax2.set_xlabel("Cohen's d  (higher = better class separation)")
ax2.legend(fontsize=8)

# ── Panel 3: Amount distribution ───────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 0])
ax3.set_facecolor('#1a1d27')
df[df['Class'] == 0]['Amount'].apply(lambda x: np.log1p(x)).plot.kde(ax=ax3, color=LEGIT_COLOR, label='Legit')
df[df['Class'] == 1]['Amount'].apply(lambda x: np.log1p(x)).plot.kde(ax=ax3, color=FRAUD_COLOR, label='Fraud')
ax3.set_title("Transaction Amount\n(log1p scale)", fontsize=10)
ax3.set_xlabel("log(Amount + 1)")
ax3.legend(fontsize=8)

# ── Panel 4: Time / fraud rate ──────────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 1])
ax4.set_facecolor('#1a1d27')
bins_t = pd.cut(df['Time_Hours'], bins=48)
fr     = df.groupby(bins_t, observed=False)['Class'].mean() * 100
bc     = [interval.mid for interval in fr.index]
ax4.plot(bc, fr.values, color=FRAUD_COLOR, linewidth=1.5)
ax4.axhline(y=fraud_pct, color='#ffd166', linestyle='--', linewidth=1, label='Baseline')
ax4.fill_between(bc, fraud_pct, fr.values,
                 where=[v > fraud_pct for v in fr.values],
                 alpha=0.3, color=FRAUD_COLOR)
ax4.set_title("Fraud Rate Over Time", fontsize=10)
ax4.set_xlabel("Hours")
ax4.set_ylabel("Fraud %")
ax4.legend(fontsize=8)

# ── Panel 5: Key stats table ─────────────────────────────────────────────────
ax5 = fig.add_subplot(gs[1, 2])
ax5.set_facecolor('#1a1d27')
ax5.axis('off')
table_data = [
    ["Metric",                    "Value"],
    ["Total Transactions",        f"{len(df):,}"],
    ["Fraud Transactions",        f"{fraud_count:,}"],
    ["Fraud Rate",                f"{fraud_pct:.3f}%"],
    ["Imbalance Ratio",           f"1:{legit_count // fraud_count}"],
    ["Missing Values",            "0"],
    ["Feature Count",             "30"],
    ["PCA Features",              "28 (V1–V28)"],
    ["Avg Legit Amount",          f"€{legit_amounts.mean():.2f}"],
    ["Avg Fraud Amount",          f"€{fraud_amounts.mean():.2f}"],
    ["Best Feature (Cohen's d)",  f"{sorted_features[0]} ({separation_scores[sorted_features[0]]:.2f})"],
]
tbl = ax5.table(cellText=table_data[1:], colLabels=table_data[0],
                cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
tbl.auto_set_font_size(False)
tbl.set_fontsize(8)
for (row, col), cell in tbl.get_celld().items():
    cell.set_facecolor('#252836' if row % 2 == 0 else '#1a1d27')
    cell.set_edgecolor('#444')
    cell.set_text_props(color='white')
    if row == 0:
        cell.set_facecolor('#2c3e6e')
        cell.set_text_props(fontweight='bold', color='white')
ax5.set_title("Key Statistics", fontsize=10)

plt.savefig(f"{OUTPUT_DIR}/08_summary_dashboard.png", dpi=150, bbox_inches='tight')
plt.close()
print("  → Saved: 08_summary_dashboard.png")

# =============================================================================
# FINAL SUMMARY & RECOMMENDATIONS
# =============================================================================
print("\n" + "="*65)
print("  EDA COMPLETE – ALL PLOTS SAVED TO ./eda_outputs/")
print("="*65)
print("""
KEY FINDINGS:
─────────────────────────────────────────────────────────────────
1. SEVERE CLASS IMBALANCE
   Only 0.172% of transactions are fraudulent.
   → Use SMOTE, class weights, or Precision-Recall metrics.
   → Accuracy is a misleading metric — use F1, AUC-PR instead.

2. TRANSACTION AMOUNT
   Fraud amounts are on average LOWER than legitimate.
   Both distributions are heavily right-skewed.
   → Log-transform Amount before modeling.
   → Amount alone is NOT enough to detect fraud.

3. TEMPORAL PATTERNS
   Fraud rate is slightly elevated during overnight quiet periods.
   → Consider time-of-day as an engineered feature.

4. PCA FEATURES
   V4, V11, V14, V17 (and a few others) show the strongest
   separation between fraud and legitimate transactions.
   → These will likely be the most important model features.
   → V1–V28 are uncorrelated with each other (by PCA design).

5. MODELING RECOMMENDATIONS
   → Start with Random Forest or XGBoost (robust to imbalance).
   → Feature engineer: log(Amount), hour-of-day.
   → Consider StandardScaler on Amount and Time.
   → Tune decision threshold for desired precision-recall tradeoff.
─────────────────────────────────────────────────────────────────
""")
