import streamlit as st
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import re
from sklearn.preprocessing import MinMaxScaler

# --- Data loading & preprocessing ---

@st.cache_data
def load_and_process_data():
    data = pd.read_csv('laptops.csv', encoding='ISO-8859-1')
    data = data[['name', 'price(in Rs.)', 'ram', 'storage', 'display(in inch)', 'rating']]
    data.dropna(inplace=True)

    # Extract raw numeric values from RAM and storage columns
    def extract_numeric(value):
        numbers = re.findall(r'\d+', str(value))
        return int(numbers[0]) if numbers else 0

    data['raw_ram'] = data['ram'].apply(extract_numeric)  # Keep raw RAM in GB
    data['raw_storage'] = data['storage'].apply(extract_numeric)  # Keep raw storage in GB
    data['raw_display'] = pd.to_numeric(data['display(in inch)'], errors='coerce')

    # Normalize numerical columns for model training
    scaler = MinMaxScaler()
    data[['price_norm', 'ram_norm', 'storage_norm', 'display_norm']] = scaler.fit_transform(
        data[['price(in Rs.)', 'raw_ram', 'raw_storage', 'raw_display']]
    )

    # Assign user_id and laptop_id for embeddings (dummy user_id = row number)
    data['user_id'] = range(len(data))
    data['laptop_id'] = range(len(data))

    # If rating column missing or invalid, create random ratings between 1-5
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
        item_emb = model.item_embedding(item_tensor)  # (1, embedding_dim)
        all_emb = model.item_embedding.weight        # (num_items, embedding_dim)
        similarities = torch.matmul(all_emb, item_emb.squeeze().T)  # (num_items, 1)

    top_indices = torch.topk(similarities.squeeze(), top_n + 1).indices.tolist()  # +1 to skip self
    top_indices = [i for i in top_indices if i != item_index][:top_n]

    return data.iloc[top_indices][['name', 'price(in Rs.)', 'raw_ram', 'raw_storage', 'raw_display', 'rating']]

# --- Streamlit UI ---

st.title("Laptop Recommendation System")

# Detailed preference inputs
st.header("Customize your laptop preferences")

cpu_options = ["Any", "Intel Core i3", "Intel Core i5", "Intel Core i7", 
               "AMD Ryzen 3", "AMD Ryzen 5", "AMD Ryzen 7"]
selected_cpu = st.selectbox("Preferred CPU Type:", cpu_options)

min_ram = st.slider("Minimum RAM (GB):", 2, 64, 8)
min_storage = st.slider("Minimum Storage (GB):", 64, 2048, 256)
storage_type = st.selectbox("Storage Type:", ["Any", "SSD", "HDD"])
min_display = st.slider("Minimum Display Size (inches):", 10.0, 18.0, 13.0)
max_price = st.number_input("Maximum Price (in Rs.):", min_value=0.0, value=100000.0)
min_rating = st.slider("Minimum User Rating:", 1, 5, 3)

def filter_laptops():
    filtered = data.copy()

    # CPU filter by searching in 'name' (case insensitive)
    if selected_cpu != "Any":
        filtered = filtered[filtered['name'].str.contains(selected_cpu, case=False)]

    # Filter by raw RAM and storage
    filtered = filtered[filtered['raw_ram'] >= min_ram]
    filtered = filtered[filtered['raw_storage'] >= min_storage]

    # Storage type filtering (approximate by name keywords)
    if storage_type != "Any":
        filtered = filtered[filtered['name'].str.contains(storage_type, case=False)]

    # Display size
    filtered = filtered[filtered['raw_display'] >= min_display]

    # Price filter (raw price)
    filtered = filtered[filtered['price(in Rs.)'] <= max_price]

    # Rating filter
    filtered = filtered[filtered['rating'] >= min_rating]

    return filtered

filtered_laptops = filter_laptops()

st.write(f"Found {filtered_laptops.shape[0]} laptops matching your preferences.")

if filtered_laptops.empty:
    st.warning("No laptops match your criteria. Please adjust your preferences.")
else:
    base_laptop = st.selectbox("Choose a base laptop for similarity recommendations:", filtered_laptops['name'].tolist())
    num_recs = st.slider("Number of recommendations:", 1, 10, 5)

    if st.button("Recommend"):
        results = recommend_laptops(base_laptop, top_n=num_recs)
        if results is None or results.empty:
            st.error("No recommendations found.")
        else:
            # Filter recommendations to be within filtered laptops only
            results = results[results['name'].isin(filtered_laptops['name'])]
            if results.empty:
                st.warning("No similar laptops found within your filter preferences.")
            else:
                st.write(f"Recommended laptops similar to **{base_laptop}**:")
                st.dataframe(results.reset_index(drop=True))
