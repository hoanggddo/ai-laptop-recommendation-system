import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import re
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
        data['rating'] = data['rating'].astype(int)

    # Normalize for AI model only
    scaler = MinMaxScaler()
    data[['price_norm','ram_norm','storage_norm','display_norm']] = scaler.fit_transform(
        data[['price(in Rs.)','raw_ram','raw_storage','raw_display']]
    )

    # Assign IDs
    data['user_id'] = range(len(data))
    data['laptop_id'] = range(len(data))
    return data

data = load_and_process_data()
num_users = data['user_id'].max()+1
num_items = data['laptop_id'].max()+1

# --- AI Model (optional) ---
class LaptopRecommendationModel(nn.Module):
    def __init__(self,num_users,num_items,embedding_dim=10):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.fc = nn.Linear(embedding_dim,1)
    def forward(self,user_ids,item_ids):
        u = self.user_embedding(user_ids)
        i = self.item_embedding(item_ids)
        return self.fc(u*i).squeeze()

model = LaptopRecommendationModel(num_users,num_items,embedding_dim=10)

def recommend_laptops(laptop_name, top_n=5):
    try:
        idx = data[data['name']==laptop_name].index[0]
        tensor_idx = torch.tensor([idx],dtype=torch.long)
        sims = model.item_embedding.weight @ model.item_embedding(tensor_idx).T
        top_idx = sims.squeeze().argsort(descending=True)[:top_n]
        return data.iloc[top_idx][['name','price_usd','raw_ram','raw_storage','raw_display','rating']]
    except:
        return pd.DataFrame()

# --- Streamlit UI ---
st.title("College Laptop Finder")
st.markdown("Find a laptop based on your needs and budget. Beginner-friendly and still customizable for advanced users.")

# User type
user_type = st.radio("Who are you?", ["High School Senior (Going to College)","I don't know much about computers","Tech-savvy user"])

# Beginner Mode
if user_type in ["High School Senior (Going to College)","I don't know much about computers"]:
    st.header("Step 1: Main Use")
    usage_type = st.selectbox("Primary Use", ["General College Work","STEM/Programming","Gaming + School","Graphic Design/Video Editing"])

    st.header("Step 2: Budget")
    max_price_usd = st.slider("Budget (USD)",300,2500,1200)
    max_price_rs = usd_to_rs(max_price_usd)

    if usage_type=="General College Work":
        min_ram,min_storage = 8,256
    else:
        min_ram,min_storage = 16,512

    min_display = 13.0
    min_rating = 3
    selected_cpu = None

    st.info(f"Recommended minimum specs:\n• RAM: {min_ram} GB\n• Storage: {min_storage} GB")

# Advanced Mode
else:
    st.header("Customize Specs")
    cpu_options = ["Any","Intel Core i3","Intel Core i5","Intel Core i7","AMD Ryzen 3","AMD Ryzen 5","AMD Ryzen 7"]
    selected_cpu = st.selectbox("CPU Preference", cpu_options)
    min_ram = st.slider("Minimum RAM (GB)",2,64,8)
    min_storage = st.slider("Minimum Storage (GB)",64,2048,256)
    min_display = st.slider("Minimum Screen Size (inches)",10.0,18.0,13.0)
    max_price_usd = st.number_input("Max Budget (USD)",0.0,2500.0,1200.0)
    max_price_rs = usd_to_rs(max_price_usd)
    min_rating = st.slider("Minimum Rating",1,5,3)
    if selected_cpu=="Any": selected_cpu=None

# Filtering
def filter_laptops():
    filtered = data.copy()
    if selected_cpu:
        filtered = filtered[filtered['name'].str.contains(selected_cpu,case=False)]
    filtered = filtered[filtered['raw_ram'] >= min_ram]
    filtered = filtered[filtered['raw_storage'] >= min_storage]
    filtered = filtered[filtered['raw_display'] >= min_display]
    filtered = filtered[filtered['price_usd'] <= max_price_usd]
    filtered = filtered[filtered['rating'] >= min_rating]
    return filtered

filtered_laptops = filter_laptops()
st.divider()

if filtered_laptops.empty:
    st.warning("No laptops match your criteria. Try adjusting your budget or requirements.")
else:
    # Prepare display df
    display_df = filtered_laptops[['name','price_usd','raw_ram','raw_storage','raw_display','rating']].copy()
    display_df['price_usd'] = display_df['price_usd'].round(2)

    # Price category
    def categorize(p):
        if p<800: return "Budget"
        elif p<1500: return "Balanced"
        else: return "Premium"
    display_df['Category'] = display_df['price_usd'].apply(categorize)

    # Score for ranking
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
        st.divider()

    st.subheader("Comparison Table")
    comp_table = top3[['name','price_usd','raw_ram','raw_storage','raw_display','rating','Category']]
    comp_table.columns = ["Model","Price (USD)","RAM (GB)","Storage (GB)","Screen Size","Rating","Category"]
    st.dataframe(comp_table)
