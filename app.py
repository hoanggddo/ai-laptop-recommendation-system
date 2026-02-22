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
def load_and_process_data():
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

    # Normalize numeric features for the model
    scaler = MinMaxScaler()
    data[['price_norm','ram_norm','storage_norm','display_norm']] = scaler.fit_transform(
        data[['price(in Rs.)','raw_ram','raw_storage','raw_display']]
    )

    # Assign IDs
    data['user_id'] = range(len(data))
    data['laptop_id'] = range(len(data))

    return data

data = load_and_process_data()

# --- Load AI model ---
with open("model_info.pkl","rb") as f:
    info = pickle.load(f)

feature_cols = info["feature_cols"]
num_users = info["num_users"]
num_items = info["num_items"]
embedding_dim = info["embedding_dim"]

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

# --- Hybrid Recommendation Function ---
def recommend_laptops(target_name, top_n=5, price_thresh=0.2, ram_thresh=0.2, storage_thresh=0.2, rating_weight=2.0):
    if target_name not in data['name'].values:
        return pd.DataFrame()

    feature_tensor = torch.tensor(data[feature_cols].values, dtype=torch.float)
    all_item_embeds = model.item_embedding.weight
    all_feature_embeds = model.fc_features(feature_tensor)

    target_idx = data[data['name']==target_name].index[0]
    target_embed = all_item_embeds[target_idx] + 0.5*all_feature_embeds[target_idx]

    similarities = F.cosine_similarity(target_embed.unsqueeze(0), all_item_embeds + 0.5*all_feature_embeds)
    top_candidates = torch.topk(similarities, top_n+5).indices.tolist()
    top_candidates = [i for i in top_candidates if i!=target_idx]

    # Feature-aware re-ranking
    filtered = []
    target_row = data.iloc[target_idx]
    for idx in top_candidates:
        row = data.iloc[idx]
        if (abs(row['price_norm'] - target_row['price_norm']) <= price_thresh and
            abs(row['ram_norm'] - target_row['ram_norm']) <= ram_thresh and
            abs(row['storage_norm'] - target_row['storage_norm']) <= storage_thresh):
            filtered.append(idx)
        if len(filtered) >= top_n*2:
            break
    if not filtered: filtered = top_candidates  # fallback

    # Compute combined score
    scores = []
    for idx in filtered:
        row = data.iloc[idx]
        score = row['rating']*rating_weight + row['ram_norm'] + row['storage_norm']
        scores.append(score)
    top_sorted = [filtered[i] for i in np.argsort(scores)[::-1][:top_n]]

    return data.iloc[top_sorted][['name','price_usd','raw_ram','raw_storage','raw_display','rating']]

# --- Streamlit UI ---
st.title("College Laptop Finder with AI Recommendations")
st.markdown("Hybrid AI + feature filtering for top laptops.")

# User type
user_type = st.radio("Who are you?", ["High School Senior","Beginner","Tech-savvy"])

if user_type in ["High School Senior","Beginner"]:
    st.header("Step 1: Main Use")
    usage_type = st.selectbox("Primary Use", ["General College Work","STEM/Programming","Gaming + School","Graphic Design/Video Editing"])
    st.header("Step 2: Budget")
    max_price_usd = st.slider("Budget (USD)",300,2500,1200)
    max_price_rs = usd_to_rs(max_price_usd)

    if usage_type=="General College Work": min_ram,min_storage=8,256
    else: min_ram,min_storage=16,512

    min_display=13.0
    min_rating=3
    selected_cpu=None
    st.info(f"Recommended minimum specs:\n• RAM: {min_ram} GB\n• Storage: {min_storage} GB")

else:
    st.header("Customize Specs")
    cpu_options=["Any","Intel Core i3","Intel Core i5","Intel Core i7","AMD Ryzen 3","AMD Ryzen 5","AMD Ryzen 7"]
    selected_cpu = st.selectbox("CPU Preference", cpu_options)
    min_ram = st.slider("Minimum RAM (GB)",2,64,8)
    min_storage = st.slider("Minimum Storage (GB)",64,2048,256)
    min_display = st.slider("Minimum Screen Size (inches)",10.0,18.0,13.0)
    max_price_usd = st.number_input("Max Budget (USD)",0.0,2500.0,1200.0)
    max_price_rs = usd_to_rs(max_price_usd)
    min_rating = st.slider("Minimum Rating",1,5,3)
    if selected_cpu=="Any": selected_cpu=None

# --- Filtering ---
def filter_laptops():
    filtered = data.copy()
    if selected_cpu: filtered = filtered[filtered['name'].str.contains(selected_cpu,case=False)]
    filtered = filtered[(filtered['raw_ram']>=min_ram) & (filtered['raw_storage']>=min_storage) &
                        (filtered['raw_display']>=min_display) & (filtered['price_usd']<=max_price_usd) &
                        (filtered['rating']>=min_rating)]
    return filtered

filtered_laptops = filter_laptops()
st.divider()

if filtered_laptops.empty:
    st.warning("No laptops match your criteria.")
else:
    display_df = filtered_laptops[['name','price_usd','raw_ram','raw_storage','raw_display','rating']].copy()
    display_df['price_usd'] = display_df['price_usd'].round(2)

    # --- AI Top Recommendations ---
    top_pick = recommend_laptops(display_df.iloc[0]['name'], top_n=3)
    st.subheader("AI Top Recommendations")
    for i,laptop in top_pick.iterrows():
        st.markdown(f"**{laptop['name']}** — ${laptop['price_usd']} | RAM: {laptop['raw_ram']} GB | Storage: {laptop['raw_storage']} GB | Screen: {laptop['raw_display']} in | Rating: {laptop['rating']}")

    st.divider()

    # --- Top 3 Pick with Badges & Progress Bars ---
    display_df['Category'] = display_df['price_usd'].apply(lambda p:"Budget" if p<800 else "Balanced" if p<1500 else "Premium")
    display_df['Score'] = display_df['rating']*2 + display_df['raw_ram']/8 + display_df['raw_storage']/256
    top3 = display_df.sort_values('Score',ascending=False).head(3).reset_index(drop=True)

    st.subheader("Top 3 Recommendations")
    for i,laptop in top3.iterrows():
        badge = "Best Overall Pick ⭐" if i==0 else ""
        color = "green" if laptop['Category']=="Budget" else "orange" if laptop['Category']=="Balanced" else "red"
        st.markdown(f"<div style='color:{color}'><strong>{laptop['name']}</strong> ({laptop['Category']})</div>",unsafe_allow_html=True)
        if badge: st.markdown(f"**{badge}**")
        st.write(f"Price: ${laptop['price_usd']}")
        st.write("RAM"); st.progress(min(laptop['raw_ram']/32,1.0))
        st.write("Storage"); st.progress(min(laptop['raw_storage']/1024,1.0))
        st.write("User Rating"); st.progress(laptop['rating']/5)
        st.write("Budget Usage"); st.progress(min(laptop['price_usd']/max_price_usd,1.0))

    st.subheader("Comparison Table")
    comp_table = top3[['name','price_usd','raw_ram','raw_storage','raw_display','rating','Category']]
    comp_table.columns = ["Model","Price (USD)","RAM (GB)","Storage (GB)","Screen Size","Rating","Category"]
    st.dataframe(comp_table)
