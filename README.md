# 💻 AI Laptop Recommendation System

An intelligent system that helps users discover laptops tailored to their preferences using a hybrid of **content filtering** and **AI-powered recommendations**. Built with **Streamlit**, **PyTorch**, and **Pandas**, it provides an interactive interface for laptop selection and similarity-based suggestions.

---

## 🚀 Overview

- **Content Filtering:** Filters laptops based on user-selected specs like RAM, storage, CPU, price, and rating.
- **AI Model:** Trained with user-item embeddings to compare laptop similarity and recommend alternatives.
- **Streamlit Interface:** Allows users to customize preferences and interact with the recommendation results.
- **Improved Performance:** Integration of user testing feedback reduced bugs and improved prediction accuracy.

---

## 🧠 Technologies Used

| Tech          | Purpose                             |
|---------------|-------------------------------------|
| Streamlit     | User Interface                      |
| PyTorch       | Recommendation model (Embeddings)   |
| Pandas        | Data manipulation                   |
| NumPy         | Numerical processing                |
| scikit-learn  | Normalization (MinMaxScaler)        |
| Regex         | Data cleaning (RAM, storage parsing)|

---

## 📁 Dataset Requirements

CSV file: `laptops.csv`  
Must include the following columns:

- `name`
- `price(in Rs.)`
- `ram`
- `storage`
- `display(in inch)`
- `rating`

---

## ⚙️ Data Preprocessing

```python
data = pd.read_csv('laptops.csv', encoding='ISO-8859-1')
data = data[['name', 'price(in Rs.)', 'ram', 'storage', 'display(in inch)', 'rating']]
data.dropna(inplace=True)

✅ Step 1: Set Up Your Project Folder
Create a folder for your project:

bash

mkdir laptop-recommender
cd laptop-recommender
Place your files in this folder:

Save the provided code as app.py

Ensure your laptop dataset (e.g., laptops.csv) is in the same folder

✅ Step 2: Set Up Your Python Environment
You need Python 3.8 or higher installed. If not installed, download from: https://www.python.org/downloads

Then, create and activate a virtual environment:

For Windows:
bash

python -m venv venv
venv\Scripts\activate
For macOS/Linux:
bash

python3 -m venv venv
source venv/bin/activate
✅ Step 3: Install Required Python Packages
Install all dependencies using pip:

bash

pip install streamlit pandas torch scikit-learn numpy
✅ Step 4: Run the App
Start your Streamlit application with:

bash

streamlit run app.py
This will open a new tab in your web browser at http://localhost:8501/
