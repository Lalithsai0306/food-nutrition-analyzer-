import warnings
warnings.filterwarnings("ignore")

import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import *
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from xgboost import XGBClassifier, XGBRegressor

st.set_page_config(page_title="Food Nutrition AI", layout="wide")

st.title("🍎 Food Nutrition: AI Analysis System")

# ============================================================
# PIPELINE
# ============================================================

@st.cache_resource(show_spinner=False)
def run_pipeline(file_bytes):
    df = pd.read_csv(io.BytesIO(file_bytes))
    df.columns = [c.strip() for c in df.columns]

    # CLEANING
    df = df.replace(["t", "t'"], 0)
    df = df.replace(",", "", regex=True)

    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except:
            pass

    df = df.fillna(df.median(numeric_only=True)).fillna("Unknown")

    # FEATURE ENGINEERING
    if {"Protein","Fiber","Calories"}.issubset(df.columns):
        df["nutrient_density"] = (df["Protein"]+df["Fiber"])/(df["Calories"]+1e-6)

    # TARGET
    q1, q2 = df["Calories"].quantile([0.33,0.66])

    def label(x):
        if x<=q1: return "Low"
        elif x<=q2: return "Medium"
        return "High"

    df["calorie_group"] = df["Calories"].apply(label)

    # ENCODING
    X = df.drop(columns=["Calories","calorie_group"], errors="ignore")
    X = pd.get_dummies(X, drop_first=True).fillna(0)

    y_cls = df["calorie_group"].astype("category").cat.codes
    y_reg = df["Calories"]

    # SPLIT
    X_train,X_test,y_train,y_test = train_test_split(X,y_cls,test_size=0.25,random_state=42)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # MODELS
    clf = RandomForestClassifier(n_estimators=120, random_state=42)
    clf.fit(X_train,y_train)

    reg = RandomForestRegressor(n_estimators=120, random_state=42)
    reg.fit(X,y_reg)

    # SIMILARITY
    sim = cosine_similarity(X)

    food_map = {}
    if "Food" in df.columns:
        food_map = {f:i for i,f in enumerate(df["Food"].astype(str))}

    return {
        "df":df,
        "X":X,
        "clf":clf,
        "reg":reg,
        "scaler":scaler,
        "sim":sim,
        "food_map":food_map
    }

# ============================================================
# INPUT PROCESSING 
# ============================================================

def prepare_input(input_dict, ctx):
    df = pd.DataFrame([input_dict])

    df = pd.get_dummies(df)

    # ALIGN columns
    for col in ctx["X"].columns:
        if col not in df.columns:
            df[col]=0

    df = df[ctx["X"].columns]

    return df

# ============================================================
# RECOMMENDER 
# ============================================================

def recommend(ctx, food, top_n=5):
    df = ctx["df"]

    if food not in ctx["food_map"]:
        matches = df[df["Food"].str.contains(food, case=False, na=False)]
        if matches.empty:
            return pd.DataFrame()
        food = matches.iloc[0]["Food"]

    idx = ctx["food_map"][food]
    sims = ctx["sim"][idx]

    top_idx = np.argsort(sims)[::-1][1:top_n+1]

    return df.iloc[top_idx][["Food","Calories"]]

# ============================================================
# UI
# ============================================================

uploaded = st.file_uploader("Upload CSV")

if uploaded:
    ctx = run_pipeline(uploaded.getvalue())

    st.success("Model Ready ✅")

    st.subheader("Prediction")

    grams = st.number_input("Grams",0.0)
    protein = st.number_input("Protein",0.0)
    fat = st.number_input("Fat",0.0)
    carbs = st.number_input("Carbs",0.0)

    if st.button("Predict"):

        data = {
            "Grams":grams,
            "Protein":protein,
            "Fat":fat,
            "Carbs":carbs
        }

        X = prepare_input(data, ctx)

        cal = ctx["reg"].predict(X)[0]
        group = ctx["clf"].predict(X)[0]

        st.success(f"Calories: {cal:.2f}")
        st.info(f"Group: {group}")

    st.subheader("Recommendation")

    food = st.text_input("Food Name")

    if st.button("Recommend"):
        rec = recommend(ctx, food)

        if rec.empty:
            st.warning("No match found")
        else:
            st.dataframe(rec)
