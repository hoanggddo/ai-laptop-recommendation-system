import streamlit as st
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import re
from sklearn.preprocessing import MinMaxScaler

# --- Currency Conversion ---
USD_RATE = 83.0  # 1 USD = 83 INR (adjust if needed)

def rs_to_usd(rs):
    return rs / USD_RATE

def usd_to_rs(usd):
    return usd * USD_RATE

# --- Data loading & preprocessing ---

@st.cache_data
def load_and_process_data():
    data = pd.read_csv('laptops.csv', encoding='ISO-8859-1')
    data = data[['name', 'price(in Rs.)', 'ram', 'storage', 'display(in inch)', 'rating']]
    data.dropna(inplace=True)

    def extract_numeric(value):
        numbers = re.findall(r'\d+', str(value))
        return int(numbers[0]) if numbers else 0

    data['raw_ram'] = data['ram'].apply(extract_numeric)
    data['raw_storage'] = data['storage'].apply(extract_numeric)
    data['raw_display'] = pd.to_numeric(data['display(in inch)'], errors='coerce')

    # Add USD price column for display
    data['price_usd'] = data['price(in Rs.)'].apply(rs_to_usd)

    scaler = MinMaxScaler()
    data[['price_norm', 'ram_norm', 'storage_norm', 'display_norm']] = scaler.fit_transform(
        data[['price(in Rs.)', 'raw_ram', 'raw_storage', 'raw_display']]
    )

    data['user_id'] = range(len(data))
    data['laptop_id'] = range(len(data))

    if data['rating'].isnull().any():
        data['rating'] = np.random.randint(1, 6, data.shape[0])
    else:
        data['rating'] = data['rating'].astype(int)

    return data

data = load_and_process_data()

num_users = data['user_id'].max() + 1
num_items = data['laptop_id'].max() + 1

# --- Model definition ---

class LaptopRecommendationModel(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=10):
        super(LaptopRecommendationModel, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.fc = nn.Linear(embedding_dim, 1)

    def forward(self, user_ids, item_ids):
        user_embeds = self.user_embedding(user_ids)
        item_embeds = self.item_embedding(item_ids)
        interaction = user_embeds * item_embeds
        score = self.fc(interaction).squeeze()
        return score

model = LaptopRecommendationModel(num_users, num_items, embedding_dim=10)

# --- Recommendation function ---

def recommend_laptops(laptop_name, top_n=5):
    if laptop_name not in data['name'].values:
        return None

    item_index = data[data['name'] == laptop_name].index[0]
    item_tensor = torch.tensor([item_index], dtype=torch.long)

    with torch.no_grad():
        item_emb = model.item_embedding(item_tensor)
        all_emb = model.item_embedding.weight
        similarities = torch.matmul(all_emb, item_emb.squeeze().T)

    top_indices = torch.topk(similarities.squeeze(), top_n + 1).indices.tolist()
    top_indices = [i for i in top_indices if i != item_index][:top_n]

    results = data.iloc[top_indices][
        ['name', 'price_usd', 'raw_ram', 'raw_storage', 'raw_display', 'rating']
    ].copy()

    results['price_usd'] = results['price_usd'].round(2)

    return results

# --- Streamlit UI ---

st.title("Laptop Recommendation System")

st.header("Customize your laptop preferences")

cpu_options = ["Any", "Intel Core i3", "Intel Core i5", "Intel Core i7",
               "AMD Ryzen 3", "AMD Ryzen 5", "AMD Ryzen 7"]
selected_cpu = st.selectbox("Preferred CPU Type:", cpu_options)

min_ram = st.slider("Minimum RAM (GB):", 2, 64, 8)
min_storage = st.slider("Minimum Storage (GB):", 64, 2048, 256)
storage_type = st.selectbox("Storage Type:", ["Any", "SSD", "HDD"])
min_display = st.slider("Minimum Display Size (inches):", 10.0, 18.0, 13.0)

# 🔥 User now inputs MAX PRICE IN USD
max_price_usd = st.number_input("Maximum Price (USD):", min_value=0.0, value=1200.0)
max_price_rs = usd_to_rs(max_price_usd)

min_rating = st.slider("Minimum User Rating:", 1, 5, 3)

def filter_laptops():
    filtered = data.copy()

    if selected_cpu != "Any":
        filtered = filtered[filtered['name'].str.contains(selected_cpu, case=False)]

    filtered = filtered[filtered['raw_ram'] >= min_ram]
    filtered = filtered[filtered['raw_storage'] >= min_storage]

    if storage_type != "Any":
        filtered = filtered[filtered['name'].str.contains(storage_type, case=False)]

    filtered = filtered[filtered['raw_display'] >= min_display]

    # Use converted rupee value internally
    filtered = filtered[filtered['price(in Rs.)'] <= max_price_rs]

    filtered = filtered[filtered['rating'] >= min_rating]

    return filtered

filtered_laptops = filter_laptops()

st.write(f"Found {filtered_laptops.shape[0]} laptops matching your preferences.")

if filtered_laptops.empty:
    st.warning("No laptops match your criteria. Please adjust your preferences.")
else:
    base_laptop = st.selectbox(
        "Choose a base laptop for similarity recommendations:",
        filtered_laptops['name'].tolist()
    )
    num_recs = st.slider("Number of recommendations:", 1, 10, 5)

    if st.button("Recommend"):
        results = recommend_laptops(base_laptop, top_n=num_recs)
        if results is None or results.empty:
            st.error("No recommendations found.")
        else:
            results = results[results['name'].isin(filtered_laptops['name'])]
            if results.empty:
                st.warning("No similar laptops found within your filter preferences.")
            else:
                st.write(f"Recommended laptops similar to **{base_laptop}**:")
                st.dataframe(results.reset_index(drop=True))
