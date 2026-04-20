# ============================================================
# Food Nutrition: Analysis and Visualization
# Streamlit Version (Optimized with Default Loading & Session State)
# ============================================================

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
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve,
    mean_absolute_error, mean_squared_error, r2_score
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from xgboost import XGBClassifier, XGBRegressor

# --- Page Config ---
st.set_page_config(page_title="Food Nutrition: Analysis and Visualization", layout="wide")

# --- Custom CSS Styling ---
st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background-color: #f5f7fa; }
[data-testid="stHeader"] { background-color: #f5f7fa; border-bottom: 1px solid #dde3ec; }
[data-testid="stSidebar"] { background-color: #eef1f7; border-right: 1px solid #d4daea; }
h1 { color: #1a2a3a !important; letter-spacing: -0.5px; border-bottom: 2px solid #c5d0e0; padding-bottom: 0.4rem; }
h2, h3 { color: #243447 !important; border-left: 4px solid #6b93c4; padding-left: 10px; }
[data-testid="stMetric"] { background: #ffffff; border: 1px solid #d0daea; border-radius: 10px; box-shadow: 0 1px 4px rgba(0,0,0,0.06); }
.stButton > button { background-color: #3a6fa8 !important; color: #ffffff !important; border-radius: 7px !important; font-weight: 600 !important; }
</style>
""", unsafe_allow_html=True)

st.title("Food Nutrition: Analysis and Visualization")
st.caption("Automated Nutritional Profiling • Machine Learning Predictions • Health Recommendations")

# --- Utility Functions ---
def rmse_compat(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

@st.cache_resource(show_spinner=False)
def run_pipeline(file_bytes):
    df = pd.read_csv(io.BytesIO(file_bytes))
    df.columns = [c.strip() for c in df.columns]
    raw_df = df.copy()

    # Data Cleaning
    df = df.replace(["t", "t'"], 0)
    df = df.replace(",", "", regex=True)
    if "Fiber" in df.columns:
        df["Fiber"] = df["Fiber"].astype(str).str.replace("a", "", regex=False)

    numeric_cols = ["Grams", "Calories", "Protein", "Fat", "Sat.Fat", "Fiber", "Carbs"]
    numeric_cols = [c for c in numeric_cols if c in df.columns]

    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Missing Value Handling (Logic Fix for New Pandas Versions)
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna("Unknown")

    # Feature Engineering
    eps = 1e-6
    if set(["Protein", "Fiber", "Calories"]).issubset(df.columns):
        df["nutrient_density"] = (df["Protein"] + df["Fiber"]) / (df["Calories"] + eps)
    if set(["Grams", "Calories"]).issubset(df.columns):
        df["calories_per_gram"] = df["Calories"] / (df["Grams"] + eps)
    
    # Classification Target (Quantile Binning)
    q1, q2 = df["Calories"].quantile(0.33), df["Calories"].quantile(0.66)
    def calorie_group(x):
        return "Low" if x <= q1 else "Medium" if x <= q2 else "High"
    df["calorie_group"] = df["Calories"].apply(calorie_group)

    # ML Prep
    drop_cols = [c for c in ["Food", "Measure"] if c in df.columns]
    X_raw = df.drop(columns=["Calories", "calorie_group"] + drop_cols).copy()
    cat_cols = X_raw.select_dtypes(exclude=[np.number]).columns.tolist()
    
    # Simple Encoding
    low_card = [c for c in cat_cols if X_raw[c].nunique() <= 10]
    high_card = [c for c in cat_cols if X_raw[c].nunique() > 10]
    freq_maps = {c: X_raw[c].value_counts(normalize=True) for c in high_card}
    for c in high_card: X_raw[c] = X_raw[c].map(freq_maps[c]).fillna(0)
    
    X = pd.get_dummies(X_raw, columns=low_card, drop_first=True).fillna(0)
    y_cls = df["calorie_group"].astype("category")
    class_names = list(y_cls.cat.categories)
    
    X_train, X_test, y_train_c, y_test_c = train_test_split(X, y_cls.cat.codes, test_size=0.25, random_state=42)
    scaler = StandardScaler().fit(X_train)

    # Models
    cls_models = {
        "Random Forest": RandomForestClassifier(n_estimators=400, random_state=42, n_jobs=1),
        "Logistic Regression": LogisticRegression(max_iter=4000, class_weight="balanced"),
        "XGBoost": XGBClassifier(n_estimators=600, random_state=42, eval_metric="mlogloss")
    }
    
    # Training (Simplified for one loop)
    results = []
    for name, model in cls_models.items():
        m_x = scaler.transform(X_train) if name == "Logistic Regression" else X_train
        model.fit(m_x, y_train_c)
        pred = model.predict(scaler.transform(X_test) if name == "Logistic Regression" else X_test)
        acc = accuracy_score(y_test_c, pred)
        results.append([name, acc])

    results_df = pd.DataFrame(results, columns=["Model", "Accuracy"]).sort_values("Accuracy", ascending=False)
    
    # Regression
    reg = XGBRegressor(n_estimators=600, random_state=42).fit(X_train, df.loc[X_train.index, "Calories"])

    # Clustering & Similarity
    c_feat = ["Calories", "Protein", "Fat", "Carbs", "Fiber"]
    c_scaled = StandardScaler().fit_transform(df[c_feat])
    df["cluster"] = KMeans(n_clusters=5, random_state=42).fit_predict(c_scaled)
    sim_mat = cosine_similarity(c_scaled)
    food_to_idx = {str(f).strip(): i for i, f in enumerate(df["Food"].astype(str).tolist())}

    return {
        "raw_df": raw_df, "df": df, "results_df": results_df, "best_model": cls_models["Random Forest"],
        "reg_model": reg, "scaler": scaler, "X": X, "class_names": class_names, "sim_mat": sim_mat,
        "food_to_idx": food_to_idx, "numeric_cols": numeric_cols, "low_card": low_card, "freq_maps": freq_maps,
        "y_test": y_test_c, "X_test": X_test, "q1": q1, "q2": q2, "cat_cols": cat_cols
    }

# --- Module Rendering Functions ---
def render_preprocessing(ctx):
    st.subheader("Data Pre-processing")
    st.write("Null handling: Median Imputation used for numerical columns.")
    st.write(f"Quantile Thresholds: Low <= {ctx['q1']:.1f}, High > {ctx['q2']:.1f}")
    st.dataframe(ctx['df'].head())

def render_eda(ctx):
    st.subheader("Exploratory Data Analysis")
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    sns.histplot(ctx['df']['Calories'], kde=True, ax=ax[0]).set_title("Calorie Distribution")
    sns.heatmap(ctx['df'][ctx['numeric_cols']].corr(), annot=True, cmap="coolwarm", ax=ax[1]).set_title("Feature Correlation")
    st.pyplot(fig)

def render_model_training(ctx):
    st.subheader("Model Performance")
    st.table(ctx['results_df'])
    st.info("XGBoost and Random Forest showed superior handling of non-linear nutrient relationships.")

def render_predictions(ctx):
    st.subheader("Nutritional Predictor")
    with st.form("pred_form"):
        c1, c2 = st.columns(2)
        u_input = {}
        fields = ["Category", "Grams", "Protein", "Fat", "Sat.Fat", "Fiber", "Carbs"]
        for i, f in enumerate(fields):
            with c1 if i%2==0 else c2:
                u_input[f] = st.text_input(f, value="0" if f != "Category" else "Dairy products")
        food_search = st.text_input("Existing Food Name (for recommendations)", value="Cows' milk")
        if st.form_submit_button("Analyze Food"):
            # Prepare row
            row = pd.DataFrame([u_input])
            for c in ctx['numeric_cols']: row[c] = pd.to_numeric(row[c], errors='coerce').fillna(0)
            # Encoding logic
            for c, fmap in ctx['freq_maps'].items(): row[c] = row[c].map(fmap).fillna(0)
            row_enc = pd.get_dummies(row, columns=[c for c in ctx['low_card'] if c in row.columns])
            for col in ctx['X'].columns: 
                if col not in row_enc.columns: row_enc[col] = 0
            row_enc = row_enc[ctx['X'].columns]
            
            cal_pred = ctx['reg_model'].predict(row_enc)[0]
            grp_pred = ctx['best_model'].predict(row_enc)[0]
            
            st.success(f"Predicted Calories: {cal_pred:.2f}")
            st.metric("Health Category", ctx['class_names'][grp_pred])

# --- Sidebar & Main Logic ---
with st.sidebar:
    st.header("Project Controls")
    uploaded_file = st.file_uploader("Upload custom dataset (Optional)", type=["csv"])
    st.markdown("---")
    module = st.radio("Navigation", ["Pre-processing", "EDA", "Model Training", "Predictions"])

# Handle Data Source
if uploaded_file:
    file_bytes = uploaded_file.getvalue()
    fid = uploaded_file.name + str(uploaded_file.size)
else:
    try:
        with open("nutrients_data.csv", "rb") as f: file_bytes = f.read()
        fid = "default_file"
        st.sidebar.info("Using system research dataset.")
    except:
        st.error("Dataset not found on server.")
        st.stop()

# Session State Cache
if "fid" not in st.session_state or st.session_state.fid != fid:
    st.session_state.fid, st.session_state.ctx = fid, None

if st.session_state.ctx is None:
    with st.spinner("⚙️ Initializing ML Pipeline..."):
        st.session_state.ctx = run_pipeline(file_bytes)

ctx = st.session_state.ctx

# Route to Module
if module == "Pre-processing": render_preprocessing(ctx)
elif module == "EDA": render_eda(ctx)
elif module == "Model Training": render_model_training(ctx)
elif module == "Predictions": render_predictions(ctx)
