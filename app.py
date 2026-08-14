"""Streamlit UI for the AI Laptop Recommender.

All data loading lives in data.py, all model logic lives in model.py.
This file only handles layout, inputs, and rendering.
"""
import pickle

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from data import compute_score, get_bar_color, load_data
from model import load_model, recommend_laptops

st.set_page_config(page_title="AI Laptop Recommender", layout="wide")


@st.cache_data
def get_data():
    return load_data("laptops.csv")


@st.cache_resource
def get_model(_feature_cols):
    with open("model_info.pkl", "rb") as f:
        info = pickle.load(f)
    model = load_model(
        "hybrid_laptop_model.pth",
        info["num_users"],
        info["num_items"],
        len(_feature_cols),
        info["embedding_dim"],
    )
    return model, info["num_items"]


data, encoders = get_data()

with open("model_info.pkl", "rb") as f:
    _model_info = pickle.load(f)
feature_cols = [c for c in _model_info["feature_cols"] if c in data.columns]
model, num_items = get_model(tuple(feature_cols))


def display_laptops(laptops, desired_specs=None, budget_usd=None):
    for _, row in laptops.iterrows():
        st.markdown(f"### {row['name']}")
        if "img_link" in row:
            st.image(row["img_link"], width=250)
        st.write(f" Price: ${row['price_usd']:.2f}")
        st.write(f" Rating: {row['rating']}/5")

        score = compute_score(row, desired_specs, budget_usd) if desired_specs and budget_usd else 0
        st.progress(score, text=f"Score: {score:.2f}")

        ram_color = get_bar_color(row["ram"], desired_specs["ram"]) if desired_specs else "green"
        storage_color = get_bar_color(row["storage"], desired_specs["storage"]) if desired_specs else "green"
        display_color = get_bar_color(row["display(in inch)"], desired_specs["display"]) if desired_specs else "green"

        fig = go.Figure()
        specs = ["RAM (GB)", "Storage (GB)", "Display (inch)"]
        values = [row["ram"], row["storage"], row["display(in inch)"]]
        colors = [ram_color, storage_color, display_color]
        desired_values = (
            [desired_specs["ram"], desired_specs["storage"], desired_specs["display"]]
            if desired_specs
            else values
        )

        fig.add_trace(go.Bar(x=specs, y=values, marker_color=colors, text=values, textposition="auto", name="Laptop Specs"))
        fig.add_trace(go.Scatter(
            x=specs, y=desired_values, mode="lines+markers",
            line=dict(color="blue", width=2, dash="dash"), name="Desired Specs",
        ))
        fig.update_layout(yaxis=dict(title="Value"), showlegend=True)
        st.plotly_chart(fig, use_container_width=True, key=f"spec_chart_{row.name}")
        st.write("---")


def display_top_summary(laptops, desired_specs=None, budget_usd=None, top_n=5):
    summary_rows = []
    for _, row in laptops.iterrows():
        score = compute_score(row, desired_specs, budget_usd) if desired_specs and budget_usd else 0
        summary_rows.append({
            "Laptop": row["name"],
            "Price ($USD)": f"${row['price_usd']:.2f}",
            "Rating": row["rating"],
            "Score": f"{score:.2f}",
        })
    summary_df = (
        pd.DataFrame(summary_rows)
        .sort_values(by="Score", ascending=False)
        .head(top_n)
    )
    st.subheader(f"Top {top_n} Recommended Laptops")
    st.table(summary_df)


def show_recommendations(ram, storage, display_val, budget_usd):
    """Shared by both Beginner and Advanced modes — was duplicated before."""
    custom_features = []
    for col in feature_cols:
        if "ram" in col.lower():
            custom_features.append(ram)
        elif "storage" in col.lower():
            custom_features.append(storage)
        elif "display" in col.lower():
            custom_features.append(display_val)
        elif "price" in col.lower():
            custom_features.append(budget_usd)
        else:
            custom_features.append(data[col].mean())

    desired_specs = {"ram": ram, "storage": storage, "display": display_val}

    recs = recommend_laptops(model, data, feature_cols, num_items, custom_features=custom_features, top_n=10)
    recs = recs.copy()
    recs["score"] = recs.apply(lambda x: compute_score(x, desired_specs, budget_usd), axis=1)
    recs = recs.sort_values(by="score", ascending=False)

    display_top_summary(recs, desired_specs, budget_usd, top_n=5)
    st.subheader("Detailed Laptop Specs")
    display_laptops(recs, desired_specs, budget_usd)


# --- Title Screen ---
st.title("AI Laptop Recommender")
st.markdown("""
Welcome! This app helps you find laptops suited for your needs.

**Beginner:** You tell us what you plan to do with your laptop, and we suggest laptops with appropriate specs.
**Advanced:** You set your desired specs and budget to find the perfect laptop for you.

All prices are displayed in USD.
""")

mode = st.radio("Choose mode:", ["Beginner", "Advanced"])

if mode == "Beginner":
    st.subheader("I don't know much about computers")
    usage = st.multiselect(
        "What will you mainly use your laptop for?",
        ["Web browsing / Office", "Gaming", "Video Editing", "Programming"],
    )
    spec_map = {
        "Web browsing / Office": {"ram": 8, "storage": 256, "display": 13},
        "Gaming": {"ram": 16, "storage": 512, "display": 15},
        "Video Editing": {"ram": 32, "storage": 1024, "display": 17},
        "Programming": {"ram": 16, "storage": 512, "display": 15},
    }
    ram = max([spec_map[u]["ram"] for u in usage]) if usage else 8
    storage = max([spec_map[u]["storage"] for u in usage]) if usage else 256
    display_val = max([spec_map[u]["display"] for u in usage]) if usage else 13
    st.write(f"Recommended specs: RAM:{ram}GB, Storage:{storage}GB, Display:{display_val}\"")
    budget_usd = st.number_input("Approx Budget ($USD)", 200, 5000, 800)

    if st.button("Recommend Laptops"):
        show_recommendations(ram, storage, display_val, budget_usd)

elif mode == "Advanced":
    st.subheader("I know what I want")
    ram = st.slider("RAM (GB)", 4, 64, 16)
    storage = st.slider("Storage (GB)", 128, 2048, 512)
    display_val = st.slider("Display (inch)", 11, 17, 15)
    budget_usd = st.number_input("Approx Budget ($USD)", 200, 5000, 1000)

    if st.button("Recommend Laptops"):
        show_recommendations(ram, storage, display_val, budget_usd)
