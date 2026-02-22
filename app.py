import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import re

# -------------------- CURRENCY --------------------
USD_RATE = 83.0  # 1 USD = 83 INR

def rs_to_usd(rs):
    return rs / USD_RATE

def usd_to_rs(usd):
    return usd * USD_RATE

# -------------------- DATA LOADING --------------------
@st.cache_data
def load_and_process_data():
    data = pd.read_csv("laptops.csv", encoding="ISO-8859-1")
    data = data[['name','price(in Rs.)','ram','storage','display(in inch)','rating']]
    data.dropna(inplace=True)

    # Extract numeric
    def extract_numeric(val):
        nums = re.findall(r'\d+', str(val))
        return int(nums[0]) if nums else 0

    data['ram'] = data['ram'].apply(extract_numeric)
    data['storage'] = data['storage'].apply(extract_numeric)
    data['display(in inch)'] = pd.to_numeric(data['display(in inch)'], errors='coerce')

    # Add USD column
    data['price_usd'] = data['price(in Rs.)'].apply(rs_to_usd).round(2)

    # Normalize features
    scaler = MinMaxScaler()
    data[['price(in Rs.)','ram','storage','display(in inch)']] = scaler.fit_transform(
        data[['price(in Rs.)','ram','storage','display(in inch)']]
    )

    # Assign IDs
    data['user_id'] = range(len(data))
    data['laptop_id'] = range(len(data))

    # If no rating column or NaNs, create random ratings
    if data['rating'].isnull().any():
        data['rating'] = np.random.randint(1,6,data.shape[0])
    else:
        data['rating'] = data['rating'].astype(int)

    return data

data = load_and_process_data()
num_users = data['user_id'].max() + 1
num_items = data['laptop_id'].max() + 1

# -------------------- MODEL --------------------
class LaptopRecommendationModel(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=10):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.fc = nn.Linear(embedding_dim, 1)

    def forward(self, user_ids, item_ids):
        user_embeds = self.user_embedding(user_ids)
        item_embeds = self.item_embedding(item_ids)
        interaction = user_embeds * item_embeds
        return self.fc(interaction).squeeze()

# Initialize model (assumes pretrained weights if available)
model = LaptopRecommendationModel(num_users, num_items, embedding_dim=10)
# model.load_state_dict(torch.load("model.pth", map_location="cpu"))  # Uncomment if saved

model.eval()

# -------------------- RECOMMENDATION FUNCTION --------------------
def recommend_laptops(base_name, top_n=3):
    try:
        item_index = data[data['name']==base_name].index[0]
        item_tensor = torch.tensor([item_index], dtype=torch.long)
        # similarity: dot product of embeddings
        similarities = model.item_embedding.weight @ model.item_embedding(item_tensor).T
        top_indices = similarities.squeeze().argsort(descending=True)[:top_n+1]  # +1 to skip self
        top_indices = [i for i in top_indices if i != item_index][:top_n]
        results = data.iloc[top_indices][['name','price_usd','ram','storage','display(in inch)','rating']].copy()
        return results
    except IndexError:
        return pd.DataFrame()

# -------------------- STREAMLIT UI --------------------
st.title("College Laptop Finder")
st.markdown("""
Find the right laptop for school or college. Beginner-friendly with advanced options for power users.
""")

# -------- USER TYPE --------
user_type = st.radio("Who are you?", ["High School Senior / Beginner",
                                      "Don't know much about computers",
                                      "Tech-savvy user"])

st.divider()

# -------- BEGINNER MODE --------
if user_type in ["High School Senior / Beginner", "Don't know much about computers"]:
    st.header("Step 1: What will you mainly use your laptop for?")
    usage_type = st.selectbox("Primary Use", [
        "General College Work (Docs, Notes, Browsing)",
        "STEM / Programming / Engineering",
        "Gaming + School",
        "Graphic Design / Video Editing"
    ])

    st.header("Step 2: Budget")
    max_price_usd = st.slider("Budget (USD)", 300, 2500, 1200)
    max_price_rs = usd_to_rs(max_price_usd)

    if usage_type == "General College Work (Docs, Notes, Browsing)":
        min_ram, min_storage = 8, 256
    else:
        min_ram, min_storage = 16, 512

    min_display = 13.0
    min_rating = 3
    selected_cpu = None

    st.info(f"Recommended minimum specs:\n• RAM: {min_ram}GB\n• Storage: {min_storage}GB")

# -------- ADVANCED MODE --------
else:
    st.header("Customize Your Specs")
    cpu_options = ["Any", "Intel Core i3","Intel Core i5","Intel Core i7",
                   "AMD Ryzen 3","AMD Ryzen 5","AMD Ryzen 7"]
    selected_cpu = st.selectbox("CPU Preference", cpu_options)
    min_ram = st.slider("Minimum RAM (GB)",2,64,8, help="RAM affects multitasking. 16GB+ recommended for heavy use.")
    min_storage = st.slider("Minimum Storage (GB)",64,2048,256, help="Storage determines file capacity. 512GB+ recommended for big programs.")
    min_display = st.slider("Minimum Screen Size (inches)",10.0,18.0,13.0)
    max_price_usd = st.number_input("Max Budget (USD)",0.0,2500.0,1200.0)
    max_price_rs = usd_to_rs(max_price_usd)
    min_rating = st.slider("Minimum Rating",1,5,3)
    if selected_cpu=="Any":
        selected_cpu=None

st.divider()

# -------- FILTER FUNCTION --------
def filter_laptops():
    filtered = data.copy()
    if selected_cpu:
        filtered = filtered[filtered['name'].str.contains(selected_cpu, case=False)]
    filtered = filtered[(filtered['ram']>=min_ram)&(filtered['storage']>=min_storage)]
    filtered = filtered[filtered['display(in inch)']>=min_display]
    filtered = filtered[filtered['price(in Rs.)']<=max_price_rs]
    filtered = filtered['rating']>=min_rating
    filtered = filtered[filtered]
    return filtered

filtered_laptops = filter_laptops()

if filtered_laptops.empty:
    st.warning("No laptops match your criteria. Try adjusting your specs or budget.")
else:
    # ---------- PREP DISPLAY ----------
    display_df = filtered_laptops[['name','price_usd','ram','storage','display(in inch)','rating']].copy()
    display_df['price_usd'] = display_df['price_usd'].round(2)

    def categorize(price):
        if price<800: return "Budget"
        elif price<1500: return "Balanced"
        else: return "Premium"
    display_df["Category"] = display_df["price_usd"].apply(categorize)
    display_df["Score"] = display_df["rating"]*2 + display_df["ram"]/8 + display_df["storage"]/256
    top3 = display_df.sort_values(by="Score", ascending=False).head(3).reset_index(drop=True)

    st.subheader("Top 3 Recommendations")
    for i in range(len(top3)):
        laptop = top3.iloc[i]
        badge = "Best Overall Pick ⭐" if i==0 else ""
        color = "green" if laptop["Category"]=="Budget" else "orange" if laptop["Category"]=="Balanced" else "red"
        st.markdown(f"<div style='color:{color}'><strong>{laptop['name']}</strong> ({laptop['Category']})</div>", unsafe_allow_html=True)
        if badge:
            st.markdown(f"**{badge}**")
        st.write(f"Price: ${laptop['price_usd']}")
        # Visual progress bars
        st.write("RAM"); st.progress(min(laptop["ram"]/32,1.0))
        st.write("Storage"); st.progress(min(laptop["storage"]/1024,1.0))
        st.write("User Rating"); st.progress(laptop["rating"]/5)
        st.write("Budget Usage"); st.progress(min(laptop["price_usd"]/max_price_usd,1.0))
        st.divider()

    # Comparison Table
    st.subheader("Compare Top 3")
    table = top3[['name','price_usd','ram','storage','display(in inch)','rating','Category']]
    table.columns = ["Model","Price(USD)","RAM(GB)","Storage(GB)","Screen Size","Rating","Category"]
    st.dataframe(table)

    # ---------- AI SIMILARITY ----------
    st.subheader("Similar Laptops Based on AI")
    base_laptop = st.selectbox("Select a base laptop for similarity recommendations", top3['name'].tolist())
    num_recs = st.slider("Number of similar recommendations",1,5,3)
    if st.button("Recommend Similar"):
        sim_results = recommend_laptops(base_laptop, top_n=num_recs)
        if sim_results.empty:
            st.warning("No similar laptops found")
        else:
            st.dataframe(sim_results[['name','price_usd','ram','storage','display(in inch)','rating']])
