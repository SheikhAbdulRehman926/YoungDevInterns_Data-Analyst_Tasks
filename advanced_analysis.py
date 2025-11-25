"""
Advanced analytics workflow for the retail KPI dataset.

Usage:
    python analysis/advanced_analysis.py

Outputs:
    outputs/clean_retail_kpis.csv
    outputs/analytics_summary.json
    outputs/model_metrics.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


DATA_PATH = Path("data/retail_kpis.csv")
OUTPUT_DIR = Path("outputs")


def _coerce_numeric(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Ensure numeric columns are numeric, coercing invalid values to NaN."""
    for column in columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def clean_data(path: Path) -> pd.DataFrame:
    """Load dataset, enforce schemas, engineer helpers, and remove duplicates."""
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    df = df.sort_values("Date")

    numeric_cols = [
        "Sales",
        "MarketingSpend",
        "NewCustomers",
        "ReturningCustomers",
        "WebTraffic",
        "CustomerSatisfaction",
    ]
    df = _coerce_numeric(df, numeric_cols)
    df = df.dropna(subset=numeric_cols)
    df = df.drop_duplicates()

    df["TotalCustomers"] = df["NewCustomers"] + df["ReturningCustomers"]
    df["NewCustomerRate"] = np.where(
        df["TotalCustomers"] > 0, df["NewCustomers"] / df["TotalCustomers"], 0
    )
    df["SalesPerMarketingDollar"] = np.where(
        df["MarketingSpend"] > 0, df["Sales"] / df["MarketingSpend"], np.nan
    )
    df["SalesPerCustomer"] = np.where(
        df["TotalCustomers"] > 0, df["Sales"] / df["TotalCustomers"], np.nan
    )
    df["Month"] = df["Date"].dt.month
    df["Quarter"] = df["Date"].dt.quarter
    return df.reset_index(drop=True)


def build_descriptive_stats(df: pd.DataFrame) -> Dict[str, Dict]:
    """Compute descriptive stats and statistical tests."""
    region_summary = (
        df.groupby("Region")["Sales"]
        .agg(["mean", "sum", "std"])
        .rename(columns={"mean": "avg_sales", "sum": "total_sales", "std": "sales_std"})
    )
    channel_summary = (
        df.groupby("Channel")["Sales"]
        .agg(["mean", "sum"])
        .rename(columns={"mean": "avg_sales", "sum": "total_sales"})
    )
    monthly_sales = df.groupby("Date")["Sales"].sum().sort_index()
    monthly_growth = monthly_sales.pct_change().fillna(0).round(4)
    corr_fields = [
        "Sales",
        "MarketingSpend",
        "WebTraffic",
        "CustomerSatisfaction",
        "NewCustomerRate",
    ]
    correlation = df[corr_fields].corr()
    linreg = stats.linregress(df["MarketingSpend"], df["Sales"])

    summary = {
        "region_summary": region_summary.round(2).to_dict(),
        "channel_summary": channel_summary.round(2).to_dict(),
        "monthly_growth": {
            idx.strftime("%Y-%m-%d"): value for idx, value in monthly_growth.items()
        },
        "correlation": correlation.round(3).to_dict(),
        "marketing_vs_sales_regression": {
            "slope": round(linreg.slope, 4),
            "intercept": round(linreg.intercept, 2),
            "r_value": round(linreg.rvalue, 4),
            "p_value": round(linreg.pvalue, 6),
            "stderr": round(linreg.stderr, 3),
        },
    }
    return summary


def train_model(df: pd.DataFrame) -> Dict[str, float]:
    """Train and evaluate a random forest regressor to predict sales."""
    target = "Sales"
    features = [
        "MarketingSpend",
        "NewCustomers",
        "ReturningCustomers",
        "WebTraffic",
        "CustomerSatisfaction",
        "Month",
        "Quarter",
        "Region",
        "Channel",
    ]
    X = df[features]
    y = df[target]

    categorical = ["Region", "Channel"]
    numeric = [col for col in features if col not in categorical]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("num", "passthrough", numeric),
        ]
    )

    model = RandomForestRegressor(
        n_estimators=300, random_state=42, max_depth=12, min_samples_leaf=2
    )

    pipeline = Pipeline(steps=[("pre", preprocessor), ("model", model)])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )
    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    metrics = {
        "mae": float(mean_absolute_error(y_test, preds)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, preds))),
        "r2": float(r2_score(y_test, preds)),
    }

    feature_names = (
        list(pipeline.named_steps["pre"].transformers_[0][1].get_feature_names_out(categorical))
        + numeric
    )
    feature_importances = pipeline.named_steps["model"].feature_importances_
    ranked_features = sorted(
        zip(feature_names, feature_importances), key=lambda x: x[1], reverse=True
    )
    metrics["top_features"] = [
        {"feature": name, "importance": round(score, 4)}
        for name, score in ranked_features[:5]
    ]

    return metrics


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    df = clean_data(DATA_PATH)
    df.to_csv(OUTPUT_DIR / "clean_retail_kpis.csv", index=False)

    stats_summary = build_descriptive_stats(df)
    (OUTPUT_DIR / "analytics_summary.json").write_text(
        json.dumps(stats_summary, indent=2)
    )

    model_metrics = train_model(df)
    (OUTPUT_DIR / "model_metrics.json").write_text(
        json.dumps(model_metrics, indent=2)
    )

    print("=== Data Preview ===")
    print(df.head())
    print("\n=== Key Insights ===")
    best_region = max(
        stats_summary["region_summary"]["total_sales"], key=stats_summary["region_summary"]["total_sales"].get
    )
    print(f"Top performing region by total sales: {best_region}")
    print(
        "Marketing spend vs sales slope:"
        f" {stats_summary['marketing_vs_sales_regression']['slope']}"
    )
    print("\n=== Model Metrics ===")
    print(json.dumps(model_metrics, indent=2))


if __name__ == "__main__":
    main()

