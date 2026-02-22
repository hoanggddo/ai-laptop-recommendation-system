import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import re

# Optional PyTorch import for later
# import torch
# import torch.nn as nn

# ---------------- Currency Conversion ----------------
USD_RATE = 83.0  # 1 USD = 83 INR
def rs_to_usd(rs): return rs / USD_RATE
def usd_to_rs(usd): return usd * USD_RATE

# ---------------- Load & Process Data ----------------
@st.cache_data
def load_data():
    data = pd.read_csv('laptops.csv', encoding='ISO-8859-1')
    data = data[['name', 'price(in Rs.)', 'ram', 'storage', 'display(in inch)', 'rating']]
    data.dropna(inplace=True)
    
    # Extract numeric
    def extract_numeric(value):
        numbers = re.findall(r'\d+', str(value))
        return int(numbers[0]) if numbers else 0
    
    data['ram'] = data['ram'].apply(extract_numeric)
    data['storage'] = data['storage'].apply(extract_numeric)
    data['display(in inch)'] = pd.to_numeric(data['display(in inch)'], errors='coerce')
    
    # Normalize numerical features
    scaler = MinMaxScaler()
    data[['price(in Rs.)','ram','storage','display(in inch)']] = scaler.fit_transform(
        data[['price(in Rs.)','ram','storage','display(in inch)']]
    )
    
    # Add USD column
    data['price_usd'] = data['price(in Rs.)'].apply(rs_to_usd)
    
    # Add dummy IDs for later AI model
    data['user_id'] = range(len(data))
    data['laptop_id'] = range(len(data))
    
    # Fill missing ratings with random integers (1-5)
    if data['rating'].isnull().any():
        data['rating'] = np.random.randint(1, 6, len(data))
    else:
        data['rating'] = data['rating'].astype(int)
    
    return data

data = load_data()

# ---------------- Streamlit UI ----------------
st.title("College Laptop Finder")
st.markdown("""
Find the right laptop for your needs and budget.
Designed for beginners, high school students, or anyone unsure about specs.
""")

# ---------------- User Type ----------------
user_type = st.radio(
    "Who are you?",
    ["High School Student / College Prep",
     "Beginner - I don't know much about computers",
     "Advanced / Tech-Savvy"]
)

st.divider()

# ---------------- Beginner / College Mode ----------------
if user_type in ["High School Student / College Prep", "Beginner - I don't know much about computers"]:
    st.header("Step 1: What will you mainly use it for?")
    usage = st.selectbox(
        "Primary Use",
        ["General College Work (Docs, Browsing)",
         "STEM / Programming / Engineering",
         "Gaming + School",
         "Graphic Design / Video Editing"]
    )
    
    st.header("Step 2: Budget")
    max_price_usd = st.slider("Max Budget (USD)", 300, 2500, 1200)
    max_price_rs = usd_to_rs(max_price_usd)
    
    # Default recommended specs
    if usage == "General College Work (Docs, Browsing)":
        min_ram, min_storage = 8, 256
    else:
        min_ram, min_storage = 16, 512
    
    min_display = 13.0
    min_rating = 3
    selected_cpu = None
    
    st.info(f"Recommended minimum specs:\n• RAM: {min_ram} GB\n• Storage: {min_storage} GB")

# ---------------- Advanced / Custom Mode ----------------
else:
    st.header("Step 1: Customize Specs")
    cpu_options = ["Any", "Intel Core i3", "Intel Core i5", "Intel Core i7",
                   "AMD Ryzen 3", "AMD Ryzen 5", "AMD Ryzen 7"]
    selected_cpu = st.selectbox("CPU Preference", cpu_options)
    
    min_ram = st.slider("Minimum RAM (GB)", 2, 64, 8,
                        help="RAM affects multitasking and speed.")
    min_storage = st.slider("Minimum Storage (GB)", 64, 2048, 256,
                            help="Storage affects file capacity and speed.")
    min_display = st.slider("Minimum Screen Size (inches)", 10.0, 18.0, 13.0)
    max_price_usd = st.number_input("Maximum Budget (USD)", 0.0, 5000.0, 1200.0)
    max_price_rs = usd_to_rs(max_price_usd)
    min_rating = st.slider("Minimum User Rating", 1, 5, 3)

    if selected_cpu == "Any": selected_cpu = None

# ---------------- Filter Function ----------------
def filter_laptops():
    filtered = data.copy()
    if selected_cpu:
        filtered = filtered[filtered['name'].str.contains(selected_cpu, case=False)]
    filtered = filtered[filtered['ram'] >= min_ram]
    filtered = filtered[filtered['storage'] >= min_storage]
    filtered = filtered[filtered['display(in inch)'] >= min_display]
    filtered = filtered[filtered['price(in Rs.)'] <= max_price_rs]
    filtered = filtered[filtered['rating'] >= min_rating]
    return filtered

filtered_laptops = filter_laptops()
st.divider()

if filtered_laptops.empty:
    st.warning("No laptops match your criteria. Try adjusting budget or specs.")
else:
    # Display top 3 only
    filtered_laptops['Score'] = filtered_laptops['rating']*2 + filtered_laptops['ram']/8 + filtered_laptops['storage']/256
    top3 = filtered_laptops.sort_values(by='Score', ascending=False).head(3).reset_index(drop=True)
    
    st.subheader("Top 3 Recommended Laptops")
    for i, row in top3.iterrows():
        badge = "Best Overall Pick ⭐" if i==0 else ""
        category = ("Budget" if row['price_usd']<800 else "Balanced" if row['price_usd']<1500 else "Premium")
        
        color = "green" if category=="Budget" else "orange" if category=="Balanced" else "red"
        st.markdown(f"<div style='color:{color}'><strong>{row['name']}</strong> ({category})</div>", unsafe_allow_html=True)
        if badge: st.markdown(f"**{badge}**")
        st.write(f"Price: ${row['price_usd']:.2f}")
        
        # Visual progress bars
        st.write("RAM"); st.progress(min(row['ram']/32,1.0))
        st.write("Storage"); st.progress(min(row['storage']/1024,1.0))
        st.write("User Rating"); st.progress(row['rating']/5)
        st.write("Budget Usage"); st.progress(min(row['price_usd']/max_price_usd,1.0))
        st.divider()
    
    st.subheader("Compare Top 3 Laptops")
    comparison_table = top3[['name','price_usd','ram','storage','display(in inch)','rating']]
    comparison_table.columns = ["Model","Price (USD)","RAM (GB)","Storage (GB)","Screen Size (inches)","Rating"]
    st.dataframe(comparison_table)

# ---------------- Placeholder for AI Model ----------------
# Later, you can add:
# import torch
# model = LaptopRecommendationModel(num_users, num_items, embedding_dim=10)
# Then integrate `recommend_laptops()` using embeddings.
