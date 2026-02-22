import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import re
import pickle
from sklearn.preprocessing import MinMaxScaler

# --- Currency Conversion ---
USD_RATE = 83.0
def rs_to_usd(rs): return rs / USD_RATE
def usd_to_rs(usd): return usd * USD_RATE

# --- Load & preprocess data ---
@st.cache_data
def load_and_process_data(feature_cols_expected):
    data = pd.read_csv('laptops.csv', encoding='ISO-8859-1')
    data = data[['name','price(in Rs.)','ram','storage','display(in inch)','rating']].dropna()

    # Extract numeric values
    def extract_numeric(val):
        nums = re.findall(r'\d+', str(val))
        return int(nums[0]) if nums else 0

    data['raw_ram'] = data['ram'].apply(extract_numeric)
    data['raw_storage'] = data['storage'].apply(extract_numeric)
    data['raw_display'] = pd.to_numeric(data['display(in inch)'], errors='coerce')
    data['price_usd'] = data['price(in Rs.)'].apply(rs_to_usd)

    # Fill missing ratings
    if data['rating'].isnull().any():
        data['rating'] = np.random.randint(1,6, len(data))
    else:
        data['rating'] = data['rating'].astype(float)

    # Ensure all feature columns exist
    for col in feature_cols_expected:
        if col not in data.columns:
            st.warning(f"Creating missing column: {col}")
            if col == 'price_norm':
                data[col] = MinMaxScaler().fit_transform(data[['price(in Rs.)']])
            elif col == 'ram_norm':
                data[col] = MinMaxScaler().fit_transform(data[['raw_ram']])
            elif col == 'storage_norm':
                data[col] = MinMaxScaler().fit_transform(data[['raw_storage']])
            elif col == 'display_norm':
                data[col] = MinMaxScaler().fit_transform(data[['raw_display']])
            else:
                st.error(f"Unknown feature column: {col}, defaulting to zeros")
                data[col] = 0.0

    # Reorder columns to match feature_cols
    data = data.copy()  # avoid SettingWithCopy warnings
    data = data.reset_index(drop=True)

    # Assign IDs
    data['user_id'] = range(len(data))
    data['laptop_id'] = range(len(data))

    return data

# --- Load model metadata ---
with open("model_info.pkl","rb") as f:
    info = pickle.load(f)

feature_cols = info["feature_cols"]
num_users = info["num_users"]
num_items = info["num_items"]
embedding_dim = info["embedding_dim"]

# --- Load data with guaranteed feature columns ---
data = load_and_process_data(feature_cols)

# --- Load AI model ---
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
