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

    if usage_type == "General College Work (Notes, Docs, Browsing)":
        min_ram, min_storage = 8, 256
    elif usage_type == "STEM / Engineering / Programming":
        min_ram, min_storage = 16, 512
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
        "Minimum RAM (GB)",
        2, 64, 8,
        help="RAM affects multitasking. 8GB is standard for school. 16GB+ is better for engineering, coding, or gaming."
    )

    min_storage = st.slider(
        "Minimum Storage (GB)",
        64, 2048, 256,
        help="Storage determines how many files/programs you can keep. 256GB is good for school. 512GB+ if you install large software or games."
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

    filtered = filtered[filtered['raw_ram'] >= min_ram]
    filtered = filtered[filtered['raw_storage'] >= min_storage]
    filtered = filtered[filtered['raw_display'] >= min_display]
    filtered = filtered[filtered['price(in Rs.)'] <= max_price_rs]
    filtered = filtered[filtered['rating'] >= min_rating]

    return filtered

filtered_laptops = filter_laptops()

st.divider()

if filtered_laptops.empty:
    st.warning("No laptops match your criteria. Try adjusting your budget or requirements.")
else:

    # Prepare display dataframe
    display_df = filtered_laptops[['name', 'price_usd', 'raw_ram',
                                    'raw_storage', 'raw_display', 'rating']].copy()

    display_df['price_usd'] = display_df['price_usd'].round(2)

    # -------- PRICE CATEGORY --------
    def categorize(price):
        if price < 800:
            return "Budget"
        elif price < 1500:
            return "Balanced"
        else:
            return "Premium"

    display_df["Category"] = display_df["price_usd"].apply(categorize)

    # -------- SCORING FOR RANKING --------
    display_df["Score"] = (
        display_df["rating"] * 2 +
        display_df["raw_ram"] / 8 +
        display_df["raw_storage"] / 256
    )

    display_df = display_df.sort_values(by="Score", ascending=False)

    # -------- TOP 3 ONLY --------
    top3 = display_df.head(3).reset_index(drop=True)

    st.subheader("Top 3 Recommended Laptops")

    for i in range(len(top3)):
        laptop = top3.iloc[i]
    
        badge = "Best Overall Pick ⭐" if i == 0 else ""
    
        # Color-coded category
        if laptop["Category"] == "Budget":
            st.markdown(f"<div style='color:green'><strong>{laptop['name']}</strong> ({laptop['Category']})</div>", unsafe_allow_html=True)
        elif laptop["Category"] == "Balanced":
            st.markdown(f"<div style='color:orange'><strong>{laptop['name']}</strong> ({laptop['Category']})</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='color:red'><strong>{laptop['name']}</strong> ({laptop['Category']})</div>", unsafe_allow_html=True)
    
        if badge:
            st.markdown(f"**{badge}**")
    
        st.write(f"Price: ${laptop['price_usd']}")
        
        # -------- VISUAL PROGRESS BARS --------
    
        # RAM bar (normalize to 32GB max for visualization)
        st.write("RAM")
        st.progress(min(laptop["raw_ram"] / 32, 1.0))
    
        # Storage bar (normalize to 1024GB)
        st.write("Storage")
        st.progress(min(laptop["raw_storage"] / 1024, 1.0))
    
        # Rating bar (5 max)
        st.write("User Rating")
        st.progress(laptop["rating"] / 5)
    
        # Price vs Budget bar (shows how close it is to budget)
        st.write("Budget Usage")
        st.progress(min(laptop["price_usd"] / max_price_usd, 1.0))
    
        st.divider()


    # -------- COMPARISON TABLE --------
    st.subheader("Compare Top 3")

    comparison_table = top3[['name', 'price_usd', 'raw_ram',
                              'raw_storage', 'raw_display', 'rating', 'Category']]

    comparison_table.columns = [
        "Model",
        "Price (USD)",
        "RAM (GB)",
        "Storage (GB)",
        "Screen Size (inches)",
        "Rating",
        "Category"
    ]

    st.dataframe(comparison_table)
