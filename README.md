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
