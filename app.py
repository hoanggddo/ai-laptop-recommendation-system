# streamlit_app.py
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import pickle
import re
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="💻 Hybrid Laptop Recommender", layout="wide")

# --- 1. Load Data ---
@st.cache_data
def load_data(csv_path="laptops.csv"):
    data = pd.read_csv(csv_path, encoding="ISO-8859-1")
    data.dropna(inplace=True)
    
    # Strip whitespace from columns
    data.columns = data.columns.str.strip()
    
    # Extract numeric from RAM and Storage
    def extract_numeric(value):
        numbers = re.findall(r'\d+', str(value))
        return int(numbers[0]) if numbers else 0

    data['ram'] = data['ram'].apply(extract_numeric)
    data['storage'] = data['storage'].apply(extract_numeric)
    data['display(in inch)'] = pd.to_numeric(data['display(in inch)'], errors='coerce')
    
    # Encode categorical columns automatically
    categorical_cols = ['processor', 'os']
    encoders = {}
    for col in categorical_cols:
        if col in data.columns:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col].astype(str))
            encoders[col] = le
    
    return data, encoders

data, encoders = load_data()

# --- 2. Load model info ---
with open("model_info.pkl", "rb") as f:
    model_info = pickle.load(f)

feature_cols = model_info["feature_cols"]
num_users = model_info["num_users"]
num_items = model_info["num_items"]
embedding_dim = model_info["embedding_dim"]

# Ensure feature columns exist
feature_cols = [col for col in feature_cols if col in data.columns]

if len(feature_cols) == 0:
    st.error("None of the feature columns exist in the CSV! Check column names.")
    st.stop()

# --- 3. Define model ---
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

# --- 4. Load trained model ---
model = HybridLaptopRecommender(num_users, num_items, len(feature_cols), embedding_dim)
model.load_state_dict(torch.load("hybrid_laptop_model.pth", map_location="cpu"))
model.eval()

# --- 5. Recommendation function ---
def recommend_laptops(target_name, top_n=5):
    if target_name not in data['name'].values:
        return pd.DataFrame(columns=['name','price(in Rs.)','ram','storage','display(in inch)','rating'])
    
    target_row = data[data['name'] == target_name].iloc[0]
    target_idx = target_row.name

    # Extract numeric features safely
    features_values = target_row[feature_cols].astype(float).values
    target_feature = torch.tensor(features_values, dtype=torch.float)
    
    all_features = torch.tensor(data[feature_cols].values.astype(float), dtype=torch.float)
    all_item_embeds = model.item_embedding.weight
    all_feature_embeds = model.fc_features(all_features)
    target_embed = all_item_embeds[target_idx] + 0.5 * all_feature_embeds[target_idx]

    similarities = F.cosine_similarity(target_embed.unsqueeze(0), all_item_embeds + 0.5 * all_feature_embeds)
    top_candidates = torch.topk(similarities, top_n + 1).indices.tolist()
    top_candidates = [i for i in top_candidates if i != target_idx][:top_n]

    return data.iloc[top_candidates][['name','price(in Rs.)','ram','storage','display(in inch)','rating']]

# --- 6. Streamlit UI ---
st.title("💻 Hybrid Laptop Recommender")

laptop_list = data['name'].unique().tolist()
selected_laptop = st.selectbox("Select a laptop to find similar ones:", laptop_list)
top_n = st.slider("Number of recommendations:", 1, 10, 5)

if st.button("Recommend"):
    recommendations = recommend_laptops(selected_laptop, top_n)
    if recommendations.empty:
        st.warning("No recommendations found.")
    else:
        st.dataframe(recommendations.reset_index(drop=True))
