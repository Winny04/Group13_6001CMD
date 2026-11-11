"""
Member 1: Boon
Part B: Exploratory Data Analysis (EDA) - 6001CMD Machine Learning
Programmatic data exploration: descriptive statistics, missing value audit,
feature distributions, outlier checks, correlation analysis, and categorical drivers.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def main():
    # 1. Ensure high-resolution outputs directory exists
    output_dir = "modules/01_eda_and_baseline/outputs"
    os.makedirs(output_dir, exist_ok=True)

    # 2. Load raw dataset
    csv_path = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    if not os.path.exists(csv_path):
        csv_path = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"

    df = pd.read_csv(csv_path)
    print("=" * 60)
    print("BOON - PART B: EXPLORATORY DATA ANALYSIS")
    print("=" * 60)
    print(f"Total Rows: {df.shape[0]}, Total Columns: {df.shape[1]}")

    # 3. Missing values audit (Identify 11 blank spaces in TotalCharges)
    df["TotalCharges_num"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    missing_count = df["TotalCharges_num"].isna().sum()
    print(f"\n[Missing Values Check] Hidden blank strings (' ') in TotalCharges: {missing_count} rows")
    blank_tenures = df[df["TotalCharges_num"].isna()]["tenure"].tolist()
    print(f"-> All these {missing_count} rows have tenure values: {set(blank_tenures)} (Structural Zeros)")

    # 4. Check non-ID duplicate records
    dups = df.drop(columns=["customerID"]).duplicated().sum()
    print(f"\n[Duplicates Check] Duplicate records when excluding customerID: {dups} rows")

    # 5. Descriptive statistics and skewness
    print("\n--- Descriptive Statistics for Numeric Features ---")
    num_cols = ["tenure", "MonthlyCharges", "TotalCharges_num"]
    stats_df = df[num_cols].describe().T
    stats_df["skewness"] = df[num_cols].skew()
    print(stats_df[["mean", "std", "min", "50%", "max", "skewness"]])

    # 6. Target class distribution (Churn)
    churn_counts = df["Churn"].value_counts()
    churn_pct = df["Churn"].value_counts(normalize=True) * 100
    print("\n--- Target Class Distribution (Churn) ---")
    print(pd.DataFrame({"Count": churn_counts, "Percentage (%)": churn_pct.round(2)}))

    # Set visualization theme
    sns.set_theme(style="whitegrid")

    # Figure 1: Target Churn distribution (Class Imbalance)
    plt.figure(figsize=(6, 4))
    ax = sns.countplot(data=df, x="Churn", hue="Churn", palette=["#2b5c8f", "#d95f02"], legend=False)
    plt.title("Figure 1: Target Churn Distribution (Class Imbalance)", fontsize=11, fontweight="bold")
    for p in ax.patches:
        ax.annotate(
            f"{int(p.get_height())} ({p.get_height()/len(df)*100:.1f}%)",
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center",
            va="bottom",
            fontsize=9,
        )
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig1_target_churn_distribution.png"), dpi=300)
    plt.close()

    # Figure 2: Numerical distributions and boxplots (IQR outlier checks)
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for i, col in enumerate(num_cols):
        # Kernel density distribution
        sns.histplot(data=df, x=col, hue="Churn", kde=True, ax=axes[0, i], palette=["#2b5c8f", "#d95f02"])
        axes[0, i].set_title(f"Distribution of {col}", fontsize=10, fontweight="bold")
        # IQR boxplot
        sns.boxplot(
            data=df,
            x="Churn",
            y=col,
            hue="Churn",
            ax=axes[1, i],
            palette=["#2b5c8f", "#d95f02"],
            legend=False,
        )
        axes[1, i].set_title(f"Boxplot (IQR Outlier Check): {col}", fontsize=10, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig2_distributions_and_outliers.png"), dpi=300)
    plt.close()

    # Figure 3: Pearson correlation heatmap
    df["Churn_num"] = df["Churn"].map({"Yes": 1, "No": 0})
    plt.figure(figsize=(7, 5))
    corr = df[["tenure", "MonthlyCharges", "TotalCharges_num", "Churn_num"]].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, cbar=True)
    plt.title("Figure 3: Pearson Correlation Heatmap", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig3_correlation_heatmap.png"), dpi=300)
    plt.close()

    # Figure 4: Key categorical drivers of churn
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    cat_drivers = ["Contract", "InternetService", "PaymentMethod"]
    for i, cat in enumerate(cat_drivers):
        sns.countplot(data=df, x=cat, hue="Churn", ax=axes[i], palette=["#2b5c8f", "#d95f02"])
        axes[i].set_title(f"Churn by {cat}", fontsize=10, fontweight="bold")
        axes[i].tick_params(axis="x", rotation=25)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig4_categorical_churn_drivers.png"), dpi=300)
    plt.close()

    print(f"\n[SUCCESS] All 4 high-resolution EDA figures saved to: {output_dir}")


if __name__ == "__main__":
    main()
