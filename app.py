import streamlit as st
import pandas as pd

# ---------------- LOAD DATA ----------------
# Make sure your dataset file is in the same folder
data = pd.read_csv("laptops.csv")

# ---------------- CLEANING ----------------

# Convert price to USD if needed (example conversion)
def usd_to_rs(usd):
    return usd * 83  # Adjust if needed

# If your dataset already has USD, skip this conversion
if "price_usd" not in data.columns:
    data["price_usd"] = data["price(in Rs.)"] / 83

# Ensure numeric columns
data["raw_ram"] = pd.to_numeric(data["raw_ram"], errors="coerce")
data["raw_storage"] = pd.to_numeric(data["raw_storage"], errors="coerce")
data["raw_display"] = pd.to_numeric(data["raw_display"], errors="coerce")
data["rating"] = pd.to_numeric(data["rating"], errors="coerce")

# ---------------- STREAMLIT UI ----------------

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
        help="RAM affects multitasking. 8GB is standard. 16GB+ is better for engineering, coding, or gaming."
    )

    min_storage = st.slider(
        "Minimum Storage (GB)",
        64, 2048, 256,
        help="Storage determines how many files/programs you can keep."
    )

    min_display = st.slider("Minimum Screen Size (inches)", 10.0, 18.0, 13.0)

    max_price_usd = st.number_input("Maximum Budget (USD)", min_value=0.0, value=1200.0)
    max_price_rs = usd_to_rs(max_price_usd)

    min_rating = st.slider("Minimum Rating", 1, 5, 3)

    if selected_cpu == "Any":
        selected_cpu = None

# ---------------- FILTER FUNCTION ----------------
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

# ---------------- RESULTS ----------------
if filtered_laptops.empty:
    st.warning("No laptops match your criteria. Try adjusting your budget or requirements.")
else:

    display_df = filtered_laptops[['name', 'price_usd', 'raw_ram',
                                    'raw_storage', 'raw_display', 'rating']].copy()

    display_df['price_usd'] = display_df['price_usd'].round(2)

    # -------- CATEGORY --------
    def categorize(price):
        if price < 800:
            return "Budget"
        elif price < 1500:
            return "Balanced"
        else:
            return "Premium"

    display_df["Category"] = display_df["price_usd"].apply(categorize)

    # -------- SCORING --------
    display_df["Score"] = (
        display_df["rating"] * 2 +
        display_df["raw_ram"] / 8 +
        display_df["raw_storage"] / 256
    )

    display_df = display_df.sort_values(by="Score", ascending=False)

    top3 = display_df.head(3).reset_index(drop=True)

    st.subheader("Top 3 Recommended Laptops")

    for i in range(len(top3)):
        laptop = top3.iloc[i]

        if i == 0:
            st.success("Best Overall Pick")

        # Color-coded title
        if laptop["Category"] == "Budget":
            st.markdown(f"<div style='color:green'><strong>{laptop['name']}</strong> (Budget)</div>", unsafe_allow_html=True)
        elif laptop["Category"] == "Balanced":
            st.markdown(f"<div style='color:orange'><strong>{laptop['name']}</strong> (Balanced)</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='color:red'><strong>{laptop['name']}</strong> (Premium)</div>", unsafe_allow_html=True)

        st.write(f"Price: ${laptop['price_usd']}")

        # -------- RAM --------
        st.write("RAM (GB)", help="RAM affects multitasking. 8GB is standard. 16GB+ is better for coding or gaming.")
        st.progress(min(laptop["raw_ram"] / 32, 1.0))
        st.caption(f"{laptop['raw_ram']} GB")

        # -------- STORAGE --------
        st.write("Storage (GB)", help="Storage determines how many files/programs you can keep.")
        st.progress(min(laptop["raw_storage"] / 1024, 1.0))
        st.caption(f"{laptop['raw_storage']} GB")

        # -------- RATING --------
        st.write("User Rating", help="Average customer rating out of 5 stars.")
        st.progress(laptop["rating"] / 5)
        st.caption(f"{laptop['rating']} / 5")

        # -------- BUDGET USAGE --------
        st.write("Budget Usage", help="Shows how much of your selected budget this laptop uses.")
        budget_ratio = laptop["price_usd"] / max_price_usd if max_price_usd > 0 else 1
        st.progress(min(budget_ratio, 1.0))
        st.caption(f"${laptop['price_usd']} of ${max_price_usd}")

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

    st.dataframe(comparison_table, use_container_width=True)
