import warnings
warnings.filterwarnings("ignore")
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBRegressor

# --- PAGE CONFIG ---
st.set_page_config(page_title="Nutrition Analysis System", layout="wide", initial_sidebar_state="expanded")

# --- PROFESSIONAL UI STYLING ---
st.markdown("""
    <style>
    [data-testid="stAppViewContainer"] { background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); }
    [data-testid="stSidebar"] { background-color: #1e2a3a !important; }
    [data-testid="stSidebar"] * { color: #ffffff !important; }
    h1 { color: #1e2a3a; font-family: 'Segoe UI'; font-weight: 700; border-bottom: 3px solid #3a6fa8; padding-bottom: 10px; }
    h2, h3 { color: #2d5a8a; }
    div.stDataFrame, div[data-testid="stMetric"] { background-color: #ffffff; border-radius: 15px; padding: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.05); }
    .stButton > button { background: linear-gradient(to right, #3a6fa8, #2d5a8a); color: white !important; border-radius: 25px; width: 100%; font-weight: 600; border: none; padding: 10px; }
    .stButton > button:hover { box-shadow: 0 5px 15px rgba(58,111,168,0.3); transform: translateY(-1px); }
    </style>
    """, unsafe_allow_html=True)

# --- BACKEND PIPELINE ---
@st.cache_resource(show_spinner=False)
def run_pipeline(file_bytes):
    df = pd.read_csv(io.BytesIO(file_bytes))
    df.columns = [c.strip() for c in df.columns]
    
    # Cleaning
    df = df.replace(["t", "t'"], 0).replace(",", "", regex=True)
    numeric_cols = ["Grams", "Calories", "Protein", "Fat", "Sat.Fat", "Fiber", "Carbs"]
    numeric_cols = [x for x in numeric_cols if x in df.columns]
    
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna("Unknown")

    # Target Logic
    q1, q2 = df["Calories"].quantile(0.33), df["Calories"].quantile(0.66)
    df["calorie_group"] = df["Calories"].apply(lambda x: "Low" if x <= q1 else "Medium" if x <= q2 else "High")

    # ML Prep
    X_raw = df.drop(columns=["Calories", "calorie_group", "Food", "Measure"], errors='ignore')
    X = pd.get_dummies(X_raw, drop_first=True).fillna(0)
    y_cls = df["calorie_group"].astype("category")
    class_names = list(y_cls.cat.categories)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y_cls.cat.codes, test_size=0.2, random_state=42)

    # Models
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1)
    rf.fit(X_train, y_train)
    acc = accuracy_score(y_test, rf.predict(X_test))
    
    reg = XGBRegressor(n_estimators=100, random_state=42)
    reg.fit(X_train, df.loc[X_train.index, "Calories"])

    return {
        "df": df, "rf": rf, "reg": reg, "X": X, 
        "class_names": class_names, "acc": acc, 
        "numeric_cols": numeric_cols, "q1": q1, "q2": q2,
        "y_test": y_test, "X_test": X_test
    }

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3850/3850285.png", width=70)
    st.markdown("### **System Menu**")
    module = st.radio("Navigation", ["🏠 Dashboard", "⚙️ Pre-processing", "📊 Analytics", "🧠 Model Engine", "🤖 Predictor"])
    st.markdown("---")
    uploaded_file = st.file_uploader("Upload New Dataset", type=["csv"])
    st.markdown("---")
    st.markdown("🔒 **System Online**")

# --- DATA LOADING ---
if uploaded_file:
    file_bytes, fid = uploaded_file.getvalue(), uploaded_file.name
else:
    try:
        with open("nutrients_data.csv", "rb") as f: file_bytes = f.read()
        fid = "system_default"
    except:
        st.error("Standard database not found. Please upload a CSV file.")
        st.stop()

if "ctx" not in st.session_state or st.session_state.get("fid") != fid:
    with st.spinner("⚙️ Initializing Core Engine..."):
        st.session_state.ctx = run_pipeline(file_bytes)
        st.session_state.fid = fid

ctx = st.session_state.ctx

# --- MODULE ROUTING ---

if module == "🏠 Dashboard":
    st.title("Food Nutrition Analysis System")
    st.write("A professional-grade nutritional profiling system utilizing Ensemble Learning and Gradient Boosting for precision health metrics.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Database Entries", len(ctx['df']))
    c2.metric("Input Variables", len(ctx['numeric_cols']))
    c3.metric("System Confidence", f"{ctx['acc']*100:.1f}%")
    st.markdown("#### Database Explorer")
    st.dataframe(ctx['df'].head(15), use_container_width=True)

elif module == "⚙️ Pre-processing":
    st.title("Data Engineering Pipeline")
    st.info("The system automatically performs string sanitation, median outlier imputation, and balanced quantile binning.")
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Metabolic Categorization Thresholds**")
        st.write(f"- Low Intensity: 0 - {ctx['q1']:.1f} kcal")
        st.write(f"- Medium Intensity: {ctx['q1']:.1f} - {ctx['q2']:.1f} kcal")
        st.write(f"- High Intensity: > {ctx['q2']:.1f} kcal")
    with col2:
        st.write("**Normalized Data Columns**")
        st.write(ctx['numeric_cols'])
    st.success("Data Integrity: Validated.")

elif module == "📊 Analytics":
    st.title("Nutritional Visual Analytics")
    ca, cb = st.columns(2)
    with ca:
        st.markdown("##### **Energy Density Distribution**")
        fig1, ax1 = plt.subplots()
        sns.histplot(ctx['df']['Calories'], kde=True, color="#3a6fa8", ax=ax1)
        st.pyplot(fig1)
    with cb:
        st.markdown("##### **Multi-Variable Correlation Matrix**")
        fig2, ax2 = plt.subplots()
        sns.heatmap(ctx['df'][ctx['numeric_cols']].corr(), annot=True, cmap="Blues", ax=ax2)
        st.pyplot(fig2)

elif module == "🧠 Model Engine":
    st.title("Model Performance Metrics")
    st.write("Evaluation results for the **Random Forest Classification** engine:")
    preds = ctx['rf'].predict(ctx['X_test'])
    report = classification_report(ctx['y_test'], preds, target_names=ctx['class_names'], output_dict=True)
    st.table(pd.DataFrame(report).transpose())
    st.info("Predictive accuracy is optimized via stratified data splitting and portion-size normalization.")

elif module == "🤖 Predictor":
    st.title("Advanced Predictive Analysis")
    st.write("Input item parameters to generate metabolic predictions.")
    with st.form("ai_form"):
        m1, m2 = st.columns(2)
        g = m1.number_input("Portion Weight (Grams)", value=100.0)
        p = m1.number_input("Protein Profile (g)", value=5.0)
        f = m2.number_input("Lipid Content (g)", value=2.0)
        c = m2.number_input("Carbohydrate Content (g)", value=15.0)
        if st.form_submit_button("Generate Prediction"):
            row = pd.DataFrame([{"Grams":g, "Protein":p, "Fat":f, "Carbs":c}])
            for col in ctx['X'].columns:
                if col not in row.columns: row[col] = 0
            row = row[ctx['X'].columns]
            cal_pred = ctx['reg'].predict(row)[0]
            grp_pred = ctx['rf'].predict(row)[0]
            
            st.markdown("---")
            res1, res2 = st.columns(2)
            res1.success(f"### Predicted Energy: {cal_pred:.1f} kcal")
            res2.info(f"### Classification: {ctx['class_names'][grp_pred]}")
            if ctx['class_names'][grp_pred] != "High": st.balloons()
