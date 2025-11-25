"""
Streamlit front-end to demonstrate the predictive model in real time.

Run with:
    streamlit run analysis/streamlit_app.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

DATA_PATH = Path("data/retail_kpis.csv")


@st.cache_data
def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Date"])
    df["TotalCustomers"] = df["NewCustomers"] + df["ReturningCustomers"]
    df["SalesPerCustomer"] = np.where(df["TotalCustomers"] > 0, df["Sales"] / df["TotalCustomers"], np.nan)
    df["Month"] = df["Date"].dt.month
    df["Quarter"] = df["Date"].dt.quarter
    return df


def _feature_columns() -> List[str]:
    return [
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


@st.cache_resource
def train_pipeline(df: pd.DataFrame) -> Pipeline:
    X = df[_feature_columns()]
    y = df["Sales"]
    categorical = ["Region", "Channel"]
    numeric = [col for col in _feature_columns() if col not in categorical]

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
    score = pipeline.score(X_test, y_test)
    return pipeline, score


def sidebar_inputs(df: pd.DataFrame) -> Dict[str, float]:
    st.sidebar.header("Adjust inputs")
    region = st.sidebar.selectbox("Region", sorted(df["Region"].unique()))
    channel = st.sidebar.selectbox("Channel", sorted(df["Channel"].unique()))

    marketing = st.sidebar.slider(
        "Marketing Spend", 20000, 60000, int(df["MarketingSpend"].median()), step=1000
    )
    new_cust = st.sidebar.slider(
        "New Customers", 200, 700, int(df["NewCustomers"].median()), step=10
    )
    returning = st.sidebar.slider(
        "Returning Customers", 500, 1200, int(df["ReturningCustomers"].median()), step=10
    )
    traffic = st.sidebar.slider(
        "Web Traffic Sessions", 20000, 70000, int(df["WebTraffic"].median()), step=1000
    )
    satisfaction = st.sidebar.slider(
        "Customer Satisfaction (1-5)", 3.5, 5.0, float(df["CustomerSatisfaction"].mean()), step=0.1
    )
    month = st.sidebar.selectbox("Month", list(range(1, 13)))
    quarter = ((month - 1) // 3) + 1

    return {
        "MarketingSpend": marketing,
        "NewCustomers": new_cust,
        "ReturningCustomers": returning,
        "WebTraffic": traffic,
        "CustomerSatisfaction": satisfaction,
        "Month": month,
        "Quarter": quarter,
        "Region": region,
        "Channel": channel,
    }


def main() -> None:
    st.set_page_config(page_title="Retail KPI Model Demo", layout="wide")
    st.title("Retail Performance Simulator")
    st.caption("Interactively explore sales predictions powered by the Random Forest model.")

    df = load_data(DATA_PATH)
    pipeline, r2 = train_pipeline(df)
    user_inputs = sidebar_inputs(df)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Sales YTD", f"${df['Sales'].sum():,.0f}")
    col2.metric("Avg Marketing Efficiency", f"{(df['Sales'].sum() / df['MarketingSpend'].sum()):.2f}x")
    col3.metric("Model R²", f"{r2:.3f}")

    st.subheader("Live Prediction")
    input_df = pd.DataFrame([user_inputs])
    predicted_sales = pipeline.predict(input_df)[0]
    st.success(f"Predicted Monthly Sales: ${predicted_sales:,.0f}")

    st.markdown("#### Scenario Comparison")
    scenario_df = (
        df.groupby(["Region", "Channel"])[["Sales", "MarketingSpend"]]
        .mean()
        .reset_index()
        .rename(columns={"Sales": "AvgSales", "MarketingSpend": "AvgMarketing"})
    )
    st.dataframe(scenario_df.style.format({"AvgSales": "${:,.0f}", "AvgMarketing": "${:,.0f}"}))

    st.markdown("#### Trend Overview")
    trend = (
        df.groupby("Date")["Sales"].sum().reset_index()
    )
    st.line_chart(trend.set_index("Date"))

    st.markdown("#### Feature Importance (training set)")
    feature_names = (
        list(pipeline.named_steps["pre"].transformers_[0][1].get_feature_names_out(["Region", "Channel"]))
        + [col for col in _feature_columns() if col not in ["Region", "Channel"]]
    )
    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": pipeline.named_steps["model"].feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    st.bar_chart(importance_df.set_index("feature"))

    st.info(
        "Use the sliders on the left to see how marketing, customer mix, and satisfaction scores "
        "change the predicted sales in real time."
    )


if __name__ == "__main__":
    main()

