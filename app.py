import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import MinMaxScaler

# --- Currency Conversion ---
USD_RATE = 83.0  # 1 USD = 83 INR

def rs_to_usd(rs):
    return rs / USD_RATE

def usd_to_rs(usd):
    return usd * USD_RATE

# --- Load and process data ---
@st.cache_data
def load_and_process_data(path='laptops.csv'):
    data = pd.read_csv(path, encoding='ISO-8859-1')
    data = data[['name', 'price(in Rs.)', 'ram', 'storage', 'display(in inch)', 'rating']]
    data.dropna(inplace=True)
    
    # Extract numeric values
    def extract_numeric(value):
        numbers = re.findall(r'\d+', str(value))
        return int(numbers[0]) if numbers else 0
    
    data['ram'] = data['ram'].apply(extract_numeric).astype('float32')
    data['storage'] = data['storage'].apply(extract_numeric).astype('float32')
    data['display(in inch)'] = pd.to_numeric(data['display(in inch)'], errors='coerce').astype('float32')
    
    # Normalize numerical features
    scaler = MinMaxScaler()
    data[['price_norm', 'ram_norm', 'storage_norm', 'display_norm']] = scaler.fit_transform(
        data[['price(in Rs.)', 'ram', 'storage', 'display(in inch)']]
    ).astype('float32')
    
    # Add USD column
    data['price_usd'] = data['price(in Rs.)'].apply(rs_to_usd).astype('float32')
    
    # Assign user/laptop IDs
    data['user_id'] = np.arange(len(data), dtype=np.int32)
    data['laptop_id'] = np.arange(len(data), dtype=np.int32)
    
    # Fill missing ratings randomly
    data['rating'] = data['rating'].fillna(np.random.randint(1, 6, len(data))).astype(np.int32)
    
    return data

data = load_and_process_data()

# --- Compute similarity (NumPy version) ---
def recommend_laptops_numpy(selected_index, top_n=5):
    """Compute similarity based on normalized features only (RAM, storage, display)."""
    features = data[['ram_norm', 'storage_norm', 'display_norm']].to_numpy()
    selected_vec = features[selected_index]
    similarities = features @ selected_vec
    # Exclude the selected laptop itself
    top_indices = np.argsort(-similarities)
    top_indices = top_indices[top_indices != selected_index][:top_n]
    results = data.iloc[top_indices][['name', 'price_usd', 'ram', 'storage', 'display(in inch)', 'rating']].copy()
    results['price_usd'] = results['price_usd'].round(2)
    return results

# --- Streamlit UI ---
st.title("College Laptop Finder")
st.markdown("""
Find the right laptop based on your needs and budget.
This tool guides beginners while still giving full control to advanced users.
""")

# ---------------- USER TYPE ----------------
user_type = st.radio(
    "What best describes you?",
    ["High School Senior (Going to College)",
     "I don't know much about computers",
     "I'm a tech-savvy user"]
)

st.divider()

# ---------------- BEGINNER MODE ----------------
if user_type in ["High School Senior (Going to College)",
                 "I don't know much about computers"]:
    
    st.header("Step 1: What will you mainly use it for?")
    usage_type = st.selectbox(
        "Primary Use",
        ["General College Work (Notes, Docs, Browsing)",
         "STEM / Engineering / Programming",
         "Gaming + School",
         "Graphic Design / Video Editing"]
    )
    
    st.header("Step 2: What's your budget?")
    max_price_usd = st.slider("Budget (USD)", 300, 2500, 1200)
    max_price_rs = usd_to_rs(max_price_usd)
    
    # Set minimum specs based on usage
    if usage_type == "General College Work (Notes, Docs, Browsing)":
        min_ram, min_storage = 8, 256
    else:
        min_ram, min_storage = 16, 512
    
    min_display = 13.0
    min_rating = 3
    selected_cpu = None
    
    st.info(f"""
Recommended minimum specs:  
• RAM: {min_ram} GB  
• Storage: {min_storage} GB
""")

# ---------------- ADVANCED MODE ----------------
else:
    st.header("Customize Your Specifications")
    
    cpu_options = ["Any", "Intel Core i3", "Intel Core i5", "Intel Core i7",
                   "AMD Ryzen 3", "AMD Ryzen 5", "AMD Ryzen 7"]
    selected_cpu = st.selectbox("CPU Preference", cpu_options)
    
    min_ram = st.slider(
        "Minimum RAM (GB)", 2, 64, 8,
        help="RAM affects multitasking. 8GB is standard for school. 16GB+ is better for STEM or gaming."
    )
    min_storage = st.slider(
        "Minimum Storage (GB)", 64, 2048, 256,
        help="Storage determines how many files/programs you can keep."
    )
    min_display = st.slider("Minimum Screen Size (inches)", 10.0, 18.0, 13.0)
    max_price_usd = st.number_input("Maximum Budget (USD)", min_value=0.0, value=1200.0)
    max_price_rs = usd_to_rs(max_price_usd)
    min_rating = st.slider("Minimum Rating", 1, 5, 3)
    
    if selected_cpu == "Any":
        selected_cpu = None

# ---------------- FILTER ----------------
def filter_laptops():
    filtered = data.copy()
    if selected_cpu:
        filtered = filtered[filtered['name'].str.contains(selected_cpu, case=False)]
    filtered = filtered[
        (filtered['ram'] >= min_ram) &
        (filtered['storage'] >= min_storage) &
        (filtered['display(in inch)'] >= min_display) &
        (filtered['price(in Rs.)'] <= max_price_rs) &
        (filtered['rating'] >= min_rating)
    ]
    return filtered

filtered_laptops = filter_laptops()
st.divider()

if filtered_laptops.empty:
    st.warning("No laptops match your criteria. Try adjusting your budget or requirements.")
else:
    # Only keep top 3
    top3_indices = filtered_laptops.index[:3]
    
    st.subheader("Top 3 Recommended Laptops")
    for i, idx in enumerate(top3_indices):
        laptop = filtered_laptops.loc[idx]
        badge = "Best Overall Pick ⭐" if i == 0 else ""
        
        # Color-coded category
        price_usd = laptop['price_usd']
        category = "Budget" if price_usd < 800 else "Balanced" if price_usd < 1500 else "Premium"
        color = "green" if category == "Budget" else "orange" if category == "Balanced" else "red"
        st.markdown(f"<div style='color:{color}'><strong>{laptop['name']}</strong> ({category})</div>", unsafe_allow_html=True)
        if badge:
            st.markdown(f"**{badge}**")
        
        st.write(f"Price: ${price_usd}")
        
        # Progress bars
        st.write("RAM")
        st.progress(min(laptop["ram"]/32,1.0))
        st.write("Storage")
        st.progress(min(laptop["storage"]/1024,1.0))
        st.write("User Rating")
        st.progress(laptop["rating"]/5)
        st.write("Budget Usage")
        st.progress(min(laptop["price_usd"]/max_price_usd,1.0))
        st.divider()
    
    # Comparison Table
    st.subheader("Compare Top 3")
    comparison_table = filtered_laptops.loc[top3_indices, ['name','price_usd','ram','storage','display(in inch)','rating']].copy()
    comparison_table.columns = ["Model","Price (USD)","RAM (GB)","Storage (GB)","Screen Size (inches)","Rating"]
    st.dataframe(comparison_table)
