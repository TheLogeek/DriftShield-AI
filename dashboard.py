import sqlite3
import json
from pathlib import Path

import numpy as np
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from database import DB_PATH, get_latest_drift_results, get_recent_drift_results


def load_inference_summary() -> pd.DataFrame:
    conn = sqlite3.connect(str(DB_PATH))
    df = pd.read_sql_query(
        "SELECT id, features_json, prediction, label, created_at "
        "FROM inference_logs ORDER BY created_at DESC LIMIT 500",
        conn,
    )
    conn.close()
    return df


def load_drift_history() -> pd.DataFrame:
    conn = sqlite3.connect(str(DB_PATH))
    df = pd.read_sql_query(
        "SELECT * FROM drift_test_results ORDER BY created_at DESC LIMIT 1000",
        conn,
    )
    conn.close()
    return df


def load_baselines() -> pd.DataFrame:
    conn = sqlite3.connect(str(DB_PATH))
    df = pd.read_sql_query("SELECT * FROM reference_baseline", conn)
    conn.close()
    return df


st.set_page_config(page_title="DriftShield AI", layout="wide")
st.title("DriftShield AI — Telemetry Dashboard")

tab1, tab2, tab3, tab4 = st.tabs(
    ["Drift Overview", "Feature Details", "Reference Baselines", "Raw Inferences"]
)

with tab1:
    st.header("Latest Drift Status")
    latest = get_latest_drift_results()
    if latest:
        rows = [dict(r) for r in latest]
        df_latest = pd.DataFrame(rows)
        col_map = {
            "feature_name": "Feature",
            "test_name": "Test",
            "p_value": "p-value",
            "corrected_p_value": "Adj. p-value",
            "significant": "Drift?",
            "drift_type": "Type",
            "created_at": "Checked At",
        }
        display = df_latest.rename(columns=col_map)[list(col_map.values())]
        display["Drift?"] = display["Drift?"].apply(lambda x: "⚠️ YES" if x else "✅ no")
        st.dataframe(display, use_container_width=True)

        sig_count = sum(1 for r in latest if r["significant"])
        st.metric("Features with Significant Drift", sig_count, delta=None)
    else:
        st.info("No drift tests have been run yet. POST some /log data and set a baseline.")

    st.header("Drift History")
    hist = load_drift_history()
    if not hist.empty:
        hist["created_at"] = pd.to_datetime(hist["created_at"])
        hist["-log10(p_value)"] = -np.log10(hist["p_value"].clip(lower=1e-300))
        fig = px.scatter(
            hist,
            x="created_at",
            y="-log10(p_value)",
            color="significant",
            hover_data=["feature_name", "p_value"],
            title="Drift Test Results Over Time",
            labels={"-log10(p_value)": "-log10(p-value)", "created_at": "Time"},
            color_discrete_map={0: "green", 1: "red"},
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No drift history yet.")

with tab2:
    st.header("Feature-Level Drift")
    hist = load_drift_history()
    if not hist.empty:
        features = hist["feature_name"].unique()
        selected = st.selectbox("Select a feature", sorted(features))
        feat_df = hist[hist["feature_name"] == selected].copy()
        feat_df["created_at"] = pd.to_datetime(feat_df["created_at"])
        feat_df = feat_df.sort_values("created_at")

        col1, col2 = st.columns(2)
        with col1:
            fig_p = px.line(
                feat_df,
                x="created_at",
                y="p_value",
                title=f"{selected} — p-value over time",
            )
            fig_p.add_hline(y=0.05, line_dash="dash", line_color="red")
            st.plotly_chart(fig_p, use_container_width=True)
        with col2:
            fig_c = px.line(
                feat_df,
                x="created_at",
                y="corrected_p_value",
                title=f"{selected} — Corrected p-value (BH)",
            )
            fig_c.add_hline(y=0.05, line_dash="dash", line_color="red")
            st.plotly_chart(fig_c, use_container_width=True)
    else:
        st.info("No drift history yet.")

with tab3:
    st.header("Reference Baselines")
    baselines = load_baselines()
    if not baselines.empty:
        for _, row in baselines.iterrows():
            with st.expander(f"{row['feature_name']} ({row['feature_type']})"):
                data = json.loads(row["baseline_json"])
                st.json(data)
                st.caption(f"Based on {row['n_samples']} samples")
    else:
        st.info("No baselines set. POST to /reference/baseline to initialize.")

with tab4:
    st.header("Recent Inferences")
    df = load_inference_summary()
    if not df.empty:
        df["created_at"] = pd.to_datetime(df["created_at"])
        st.dataframe(df, use_container_width=True)

        if "label" in df.columns and df["label"].notna().any():
            labeled = df.dropna(subset=["label"])
            accuracy = (labeled["prediction"] == labeled["label"]).mean()
            st.metric("Labeled Accuracy", f"{accuracy:.2%}")
    else:
        st.info("No inference data yet.")
