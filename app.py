# streamlit_app_with_score.py
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import pickle
import re
from sklearn.preprocessing import LabelEncoder
import plotly.graph_objects as go

st.set_page_config(page_title="💻 Laptop Recommender", layout="wide")

INR_TO_USD = 0.012

# --- Load Data ---
@st.cache_data
def load_data(csv_path="laptops.csv"):
    data = pd.read_csv(csv_path, encoding="ISO-8859-1")
    data.dropna(inplace=True)
    data.columns = data.columns.str.strip()
    
    data['ram'] = data['ram'].apply(lambda x: int(re.findall(r'\d+', str(x))[0]) if re.findall(r'\d+', str(x)) else 0)
    data['storage'] = data['storage'].apply(lambda x: int(re.findall(r'\d+', str(x))[0]) if re.findall(r'\d+', str(x)) else 0)
    data['display(in inch)'] = pd.to_numeric(data['display(in inch)'], errors='coerce')
    
    categorical_cols = ['processor', 'os']
    encoders = {}
    for col in categorical_cols:
        if col in data.columns:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col].astype(str))
            encoders[col] = le

    data['price_usd'] = data['price(in Rs.)'] * INR_TO_USD
    return data, encoders

data, encoders = load_data()

# --- Load model info ---
with open("model_info.pkl", "rb") as f:
    model_info = pickle.load(f)

feature_cols = [col for col in model_info["feature_cols"] if col in data.columns]
num_users = model_info["num_users"]
num_items = model_info["num_items"]
embedding_dim = model_info["embedding_dim"]

# --- Model ---
class HybridLaptopRecommender(nn.Module):
    def __init__(self, num_users, num_items, num_features, embedding_dim=30):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.fc_features = nn.Linear(num_features, embedding_dim)
        self.fc = nn.Linear(embedding_dim, 1)
    def forward(self, user_ids, item_ids, features):
        user_embeds = self.user_embedding(user_ids)
        item_embeds = self.item_embedding(item_ids)
        feature_embeds = self.fc_features(features)
        interaction = user_embeds * (item_embeds + feature_embeds)
        return self.fc(interaction).squeeze()

model = HybridLaptopRecommender(num_users, num_items, len(feature_cols), embedding_dim)
model.load_state_dict(torch.load("hybrid_laptop_model.pth", map_location="cpu"))
model.eval()

# --- Recommendation ---
def recommend_laptops(target_name=None, top_n=5, custom_features=None):
    all_features = torch.tensor(data[feature_cols].iloc[:num_items].values.astype(float), dtype=torch.float)
    all_item_embeds = model.item_embedding.weight
    all_feature_embeds = model.fc_features(all_features)

    if target_name:
        target_row = data[data['name']==target_name].iloc[0]
        target_idx = target_row.name
        target_embed = all_item_embeds[target_idx] + 0.5*all_feature_embeds[target_idx]
    elif custom_features is not None:
        custom_features_tensor = torch.tensor(custom_features, dtype=torch.float)
        target_embed = 0.5 * model.fc_features(custom_features_tensor)
    else:
        return pd.DataFrame()
    
    similarities = F.cosine_similarity(target_embed.unsqueeze(0), all_item_embeds + 0.5*all_feature_embeds)
    top_candidates = torch.topk(similarities, top_n*3).indices.tolist()
    return data.iloc[top_candidates]

# --- Spec bar colors ---
def get_bar_color(value, desired):
    ratio = value / max(desired,1)
    if ratio >= 1.0:
        return "green"
    elif ratio >= 0.8:
        return "yellow"
    else:
        return "red"

# --- Overall Score ---
def compute_score(row, desired_specs, budget_usd):
    spec_score = 0
    for k in ['ram', 'storage', 'display(in inch)']:
        spec_score += min(row[k]/max(desired_specs[k],1),1)
    spec_score /= 3
    rating_score = row['rating']/5
    budget_score = 1 - abs(row['price_usd'] - budget_usd)/max(budget_usd,1)
    total_score = 0.5*spec_score + 0.3*rating_score + 0.2*budget_score
    return total_score

# --- Display ---
def display_laptops(laptops, desired_specs=None, budget_usd=None):
    for i, row in laptops.iterrows():
        st.markdown(f"### {row['name']}")
        if 'img_link' in row:
            st.image(row['img_link'], width=250)
        st.write(f"💰 Price: ${row['price_usd']:.2f}")
        st.write(f"⭐ Rating: {row['rating']}/5")

        # Overall Score
        score = compute_score(row, desired_specs, budget_usd) if desired_specs and budget_usd else 0
        st.progress(score)

        # Spec chart
        ram_color = get_bar_color(row['ram'], desired_specs['ram']) if desired_specs else "green"
        storage_color = get_bar_color(row['storage'], desired_specs['storage']) if desired_specs else "green"
        display_color = get_bar_color(row['display(in inch)'], desired_specs['display']) if desired_specs else "green"

        fig = go.Figure()
        specs = ['RAM (GB)','Storage (GB)','Display (inch)']
        values = [row['ram'],row['storage'],row['display(in inch)']]
        colors = [ram_color,storage_color,display_color]
        desired_values = [desired_specs['ram'],desired_specs['storage'],desired_specs['display']] if desired_specs else values

        fig.add_trace(go.Bar(x=specs, y=values, marker_color=colors, text=values, textposition='auto', name='Laptop Specs'))
        fig.add_trace(go.Scatter(x=specs, y=desired_values, mode='lines+markers', line=dict(color='blue',width=2,dash='dash'), name='Desired Specs'))
        fig.update_layout(yaxis=dict(title='Value'), showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
        st.write("---")

# --- UI ---
mode = st.radio("Choose mode:", ["Beginner","Advanced"])

if mode=="Beginner":
    st.subheader("I don’t know much about computers")
    usage = st.multiselect("What will you mainly use your laptop for?", ["Web browsing / Office", "Gaming", "Video Editing", "Programming"])
    spec_map = {
        "Web browsing / Office":{"ram":8,"storage":256,"display":13},
        "Gaming":{"ram":16,"storage":512,"display":15},
        "Video Editing":{"ram":32,"storage":1024,"display":17},
        "Programming":{"ram":16,"storage":512,"display":15}
    }
    ram = max([spec_map[u]["ram"] for u in usage]) if usage else 8
    storage = max([spec_map[u]["storage"] for u in usage]) if usage else 256
    display_val = max([spec_map[u]["display"] for u in usage]) if usage else 13
    st.write(f"Recommended specs: RAM:{ram}GB, Storage:{storage}GB, Display:{display_val}\"")
    budget_usd = st.number_input("Approx Budget ($USD)", 200, 5000, 800)

    custom_features = []
    for col in feature_cols:
        if "ram" in col.lower(): custom_features.append(ram)
        elif "storage" in col.lower(): custom_features.append(storage)
        elif "display" in col.lower(): custom_features.append(display_val)
        else: custom_features.append(data[col].mean())
    
    if st.button("Recommend Laptops"):
        recs = recommend_laptops(custom_features=custom_features, top_n=5)
        recs['score'] = recs.apply(lambda x: compute_score(x, {'ram':ram,'storage':storage,'display':display_val}, budget_usd), axis=1)
        recs = recs.sort_values(by='score', ascending=False)
        st.subheader("Top Laptop Recommendations")
        display_laptops(recs, {'ram':ram,'storage':storage,'display':display_val}, budget_usd)

elif mode=="Advanced":
    st.subheader("I know what I want")
    ram = st.slider("RAM (GB)",4,64,16)
    storage = st.slider("Storage (GB)",128,2048,512)
    display_val = st.slider("Display (inch)",11,17,15)
    budget_usd = st.number_input("Approx Budget ($USD)",200,5000,1000)

    custom_features = []
    for col in feature_cols:
        if "ram" in col.lower(): custom_features.append(ram)
        elif "storage" in col.lower(): custom_features.append(storage)
        elif "display" in col.lower(): custom_features.append(display_val)
        elif "price" in col.lower(): custom_features.append(budget_usd)
        else: custom_features.append(data[col].mean())
    
    if st.button("Recommend Laptops"):
        recs = recommend_laptops(custom_features=custom_features, top_n=5)
        recs['score'] = recs.apply(lambda x: compute_score(x, {'ram':ram,'storage':storage,'display':display_val}, budget_usd), axis=1)
        recs = recs.sort_values(by='score', ascending=False)
        st.subheader("Top Laptop Recommendations")
        display_laptops(recs, {'ram':ram,'storage':storage,'display':display_val}, budget_usd)
