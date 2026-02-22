# AI-Powered College Laptop Finder

This is a **Streamlit-based AI laptop recommendation system** designed to help users find laptops tailored to their needs, budget, and preferred specifications. It combines **feature-based filtering** with a **hybrid AI recommendation model** using PyTorch embeddings.

The app is beginner-friendly while still allowing advanced users to customize CPU, RAM, storage, display size, and budget. Top laptops are ranked using a **combined score of specs and user ratings**.

---

## Key Features

- **Beginner Mode:**  
  - Select your primary use (general work, STEM/programming, gaming, or graphics/video editing)  
  - Slider-based budget and automatic recommended specs (RAM, storage)  

- **Advanced Mode:**  
  - Full customization of CPU, RAM, storage, screen size, rating, and budget  
  - Allows precise filtering for tech-savvy users  

- **Hybrid AI Recommendations:**  
  - User-item embedding model trained with PyTorch  
  - Similar laptops recommended based on both embeddings and normalized specs  
  - Top 3 recommendations highlighted with badges (Best Overall Pick)  

- **Interactive Visual Feedback:**  
  - Progress bars for RAM, Storage, Rating, and Budget usage  
  - Laptops categorized as Budget, Balanced, or Premium  
  - Comparison table for top picks  

---

## Technologies Used

| Technology       | Purpose                                                                 |
|-----------------|-------------------------------------------------------------------------|
| Streamlit        | Web app interface, sliders, tables, progress bars                        |
| PyTorch          | Hybrid AI model for laptop similarity recommendations                    |
| Pandas           | Data manipulation and filtering                                          |
| NumPy            | Numerical processing, scoring                                           |
| scikit-learn     | Feature normalization using `MinMaxScaler`                               |
| Regex            | Extract numeric values from RAM and storage columns                      |

---

## Dataset Requirements

File: `laptops.csv`  
Columns needed:

- `name` – Laptop name  
- `price(in Rs.)` – Price in Indian Rupees  
- `ram` – RAM (e.g., "8 GB")  
- `storage` – Storage (e.g., "512 GB SSD")  
- `display(in inch)` – Screen size  
- `rating` – User rating (1–5)

The app automatically converts RAM, storage, and display to numeric values and normalizes them for the AI model.

---

## Data Preprocessing

1. **Load CSV and clean data:** Remove missing rows and extract numeric values from RAM/storage.  
2. **Add IDs for embeddings:** Each laptop gets a `laptop_id` and each row gets a `user_id`.  
3. **Normalize numeric features:** Price, RAM, storage, and display are scaled between 0 and 1.  
4. **Price conversion:** Optionally convert INR to USD for display.  

Example:

```python
import re
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv('laptops.csv', encoding='ISO-8859-1')
data = data[['name', 'price(in Rs.)', 'ram', 'storage', 'display(in inch)', 'rating']]
data.dropna(inplace=True)

def extract_numeric(val):
    nums = re.findall(r'\d+', str(val))
    return int(nums[0]) if nums else 0

data['raw_ram'] = data['ram'].apply(extract_numeric)
data['raw_storage'] = data['storage'].apply(extract_numeric)
data['raw_display'] = pd.to_numeric(data['display(in inch)'], errors='coerce')
data['price_usd'] = data['price(in Rs.)'] / 83.0  # Example conversion rate INR->USD

scaler = MinMaxScaler()
data[['price_norm','ram_norm','storage_norm','display_norm']] = scaler.fit_transform(
    data[['price(in Rs.)','raw_ram','raw_storage','raw_display']]
)

data['user_id'] = range(len(data))
data['laptop_id'] = range(len(data))

Step 1: Set Up Your Project Folder
Create a folder for your project:

bash

mkdir laptop-recommender
cd laptop-recommender
Place your files in this folder:
app.py
laptops.csv
hybrid_laptop_model.pth
model_info.pkl
requirements.txt

Ensure your laptop dataset (e.g., laptops.csv) is in the same folder

Step 2: Set Up Your Python Environment
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
Step 3: Install Required Python Packages

Create a requirements.txt with:

streamlit==1.31.0
torch==2.1.0
pandas==2.1.1
numpy==1.26.2
scikit-learn==1.3.0

Install all dependencies using pip:

bash


pip install streamlit pandas torch scikit-learn numpy
Step 4: Run the App
Start your Streamlit application with:

bash

streamlit run app.py
This will open a new tab in your web browser at http://localhost:8501/
