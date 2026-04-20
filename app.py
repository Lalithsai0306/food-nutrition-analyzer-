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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
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
    h2, h3 { color: #2d5a8a; border-left: 5px solid #3a6fa8; padding-left: 10px; }
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
    
    # Cleaning Logic
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

    # Grouping Logic
    q1, q2 = df["Calories"].quantile(0.33), df["Calories"].quantile(0.66)
    df["calorie_group"] = df["Calories"].apply(lambda x: "Low" if x <= q1 else "Medium" if x <= q2 else "High")

    # Feature Prep
    drop_cols = [c for c in ["Food", "Measure"] if c in df.columns]
    X_raw = df.drop(columns=["Calories", "calorie_group"] + drop_cols, errors='ignore').copy()
    
    cat_cols = X_raw.select_dtypes(exclude=[np.number]).columns.tolist()
    low_card = [c for c in cat_cols if X_raw[c].nunique() <= 10]
    high_card = [c for c in cat_cols if X_raw[c].nunique() > 10]
    
    freq_maps = {c: X_raw[c].value_counts(normalize=True) for c in high_card}
    for c in high_card: X_raw[c] = X_raw[c].map(freq_maps[c]).fillna(0)
    
    X = pd.get_dummies(X_raw, columns=low_card, drop_first=True).fillna(0)
    y_cls = df["calorie_group"].astype("category")
    class_names = list(y_cls.cat.categories)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y_cls.cat.codes, test_size=0.2, random_state=42)

    # Models (Speed Optimized)
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1)
    rf.fit(X_train, y_train)
    acc = accuracy_score(y_test, rf.predict(X_test))
    
    reg = XGBRegressor(n_estimators=100, random_state=42)
    reg.fit(X_train, df.loc[X_train.index, "Calories"])

    return {
        "df": df, "rf": rf, "reg": reg, "X": X, 
        "class_names": class_names, "acc": acc, 
        "numeric_cols": numeric_cols, "q1": q1, "q2": q2,
        "y_test": y_test, "X_test": X_test, "low_card": low_card, "freq_maps": freq_maps
    }

# --- SIDEBAR NAVIGATION ---
with st.sidebar:
    st.markdown("### **System Navigation**")
    module = st.radio("Pipeline Stages", ["🏠 Dashboard", "⚙️ Pre-processing", "📊 Analytics (EDA)", "🧠 Model Training", "🤖 AI Predictor"])
    st.markdown("---")
    uploaded_file = st.file_uploader("📂 Upload Custom CSV (Optional)", type=["csv"])
    st.markdown("---")
    st.caption("All-in-one Analysis Suite")

# --- DATA LOADING (DEFAULT VS UPLOADED) ---
if uploaded_file:
    file_bytes, fid = uploaded_file.getvalue(), uploaded_file.name
else:
    try:
        with open("nutrients_data.csv", "rb") as f: file_bytes = f.read()
        fid = "system_default_v2"
        st.sidebar.info("Using pre-loaded dataset.")
    except:
        st.error("Please upload nutrients_data.csv to continue.")
        st.stop()

if "ctx" not in st.session_state or st.session_state.get("fid") != fid:
    with st.spinner("⚙️ Powering up Analysis Engines..."):
        st.session_state.ctx = run_pipeline(file_bytes)
        st.session_state.fid = fid

ctx = st.session_state.ctx

# --- MODULE ROUTING ---

if module == "🏠 Dashboard":
    st.title("Nutritional Analysis System")
    st.write("This dashboard provides an end-to-end view of food nutritional density using machine learning.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Dataset Size", len(ctx['df']))
    c2.metric("Predictors", len(ctx['numeric_cols']))
    c3.metric("System Accuracy", f"{ctx['acc']*100:.1f}%")
    st.markdown("#### **Data Overview**")
    st.dataframe(ctx['df'].head(15), use_container_width=True)

elif module == "⚙️ Pre-processing":
    st.title("Data Engineering & Cleaning")
    st.info("The system applies median imputation for nulls and quantile-based labeling for health groups.")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Classification Bins")
        st.write(f"🟢 **Low**: 0 - {ctx['q1']:.1f} kcal")
        st.write(f"🟡 **Medium**: {ctx['q1']:.1f} - {ctx['q2']:.1f} kcal")
        st.write(f"🔴 **High**: > {ctx['q2']:.1f} kcal")
    with col2:
        st.subheader("Feature Normalization")
        st.write("Numerical values scaled and standardized for processing.")
        st.write(", ".join(ctx['numeric_cols']))

elif module == "📊 Analytics (EDA)":
    st.title("Exploratory Data Analysis")
    st.write("Visualizing the relationship between macronutrients and energy density.")
    ca, cb = st.columns(2)
    with ca:
        st.markdown("##### **Calorie Density Distribution**")
        fig1, ax1 = plt.subplots()
        sns.histplot(ctx['df']['Calories'], kde=True, color="#3a6fa8", ax=ax1)
        st.pyplot(fig1)
    with cb:
        st.markdown("##### **Macronutrient Correlation Heatmap**")
        fig2, ax2 = plt.subplots()
        sns.heatmap(ctx['df'][ctx['numeric_cols']].corr(), annot=True, cmap="Blues", ax=ax2)
        st.pyplot(fig2)

elif module == "🧠 Model Training":
    st.title("Model Training & Evaluation")
    st.write("Comprehensive classification report for our **Random Forest** engine.")
    preds = ctx['rf'].predict(ctx['X_test'])
    report = classification_report(ctx['y_test'], preds, target_names=ctx['class_names'], output_dict=True)
    st.table(pd.DataFrame(report).transpose())
    
    st.markdown("##### **Confusion Matrix**")
    fig3, ax3 = plt.subplots(figsize=(6,3))
    sns.heatmap(confusion_matrix(ctx['y_test'], preds), annot=True, fmt='d', cmap="Blues", ax=ax3)
    st.pyplot(fig3)

elif module == "🤖 AI Predictor":
    st.title("Predictive Nutrition Calculator")
    st.write("Fill in all parameters to generate a precise metabolic prediction.")
    with st.form("pro_form"):
        r1_c1, r1_c2, r1_c3 = st.columns(3)
        cat = r1_c1.text_input("Food Category", value="Dairy products")
        grams = r1_c2.number_input("Portion (Grams)", value=100.0)
        prot = r1_c3.number_input("Protein (g)", value=5.0)
        
        r2_c1, r2_c2, r2_c3 = st.columns(3)
        fat = r2_c1.number_input("Total Fat (g)", value=2.0)
        sfat = r2_c2.number_input("Sat. Fat (g)", value=1.0)
        fiber = r2_c3.number_input("Fiber (g)", value=0.0)
        
        carbs = st.number_input("Carbohydrates (g)", value=10.0)
        
        if st.form_submit_button("🚀 Execute Prediction"):
            # Logic to match training features exactly
            input_dict = {"Category":cat, "Grams":grams, "Protein":prot, "Fat":fat, "Sat.Fat":sfat, "Fiber":fiber, "Carbs":carbs}
            row = pd.DataFrame([input_dict])
            
            # Handling Encoding for prediction row
            for c, fmap in ctx['freq_maps'].items(): 
                if c in row.columns: row[c] = row[c].map(fmap).fillna(0)
            
            row_enc = pd.get_dummies(row, columns=[c for c in ctx['low_card'] if c in row.columns])
            for col in ctx['X'].columns:
                if col not in row_enc.columns: row_enc[col] = 0
            row_enc = row_enc[ctx['X'].columns]
            
            cal_pred = ctx['reg'].predict(row_enc)[0]
            grp_pred = ctx['rf'].predict(row_enc)[0]
            
            st.markdown("---")
            res1, res2 = st.columns(2)
            res1.success(f"### **Estimated: {cal_pred:.1f} kcal**")
            res2.info(f"### **Category: {ctx['class_names'][grp_pred]}**")
            
            if ctx['class_names'][grp_pred] == "Low": st.balloons()
