import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# (Keep the rest of your imports here...)

# 1. Page Title
st.title("🍎 Food Nutrition Analysis & Prediction System")

# 2. Load the Data FIRST
csv_file = "nutrients_data.csv"
df = pd.read_csv(csv_file)

# Clean column names just like in your original script
df.columns = [c.strip() for c in df.columns]

# 3. NOW you can display it!
st.subheader("Dataset Overview")
st.dataframe(df.head()) 
