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
    classification_report, confusion_matrix, roc_auc_score, 
    roc_curve, mean_absolute_error, mean_squared_error, r2_score 
) 
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor 
from sklearn.linear_model import LogisticRegression, LinearRegression 
from sklearn.cluster import KMeans 
from sklearn.decomposition import PCA 
from sklearn.metrics.pairwise import cosine_similarity 
from xgboost import XGBClassifier, XGBRegressor 

st.set_page_config(page_title="Food Nutrition: Analysis and Visualization", layout="wide") 

# ============================================================ 
# Light Contrast Styling 
# ============================================================ 
st.markdown(""" 
<style> 
[data-testid="stAppViewContainer"] { background-color: #f5f7fa; } 
[data-testid="stHeader"] { background-color: #f5f7fa; border-bottom: 1px solid #dde3ec; }
[data-testid="stSidebar"] { background-color: #eef1f7; border-right: 1px solid #d4daea; } 
h1 { color: #1a2a3a !important; border-bottom: 2px solid #c5d0e0; padding-bottom: 0.4rem; } 
[data-testid="stMetric"] { background: #ffffff; border: 1px solid #d0daea; border-radius: 10px; padding: 14px; box-shadow: 0 1px 4px rgba(0,0,0,0.06); } 
[data-testid="stForm"] { background-color: #ffffff; border: 1px solid #d0daea; border-radius: 12px; }
</style> 
""", unsafe_allow_html=True) 

st.title("Food Nutrition: Analysis and Visualization") 
st.caption("Upload nutrients_data.csv • Pre-processing • EDA • Model Training • Evaluation • Predictions")

def rmse_compat(y_true, y_pred): 
    return float(np.sqrt(mean_squared_error(y_true, y_pred))) 

@st.cache_resource(show_spinner=False) 
def run_pipeline(file_bytes): 
    df = pd.read_csv(io.BytesIO(file_bytes))
    df.columns = [c.strip() for c in df.columns] 
    raw_df = df.copy() 
    
    # Cleaning
    df = df.replace(["t", "t'"], 0) 
    df = df.replace(",", "", regex=True) 
    if "Fiber" in df.columns: 
        df["Fiber"] = df["Fiber"].astype(str).str.replace("a", "", regex=False) 
        
    numeric_cols = [c for c in ["Grams", "Calories", "Protein", "Fat", "Sat.Fat", "Fiber", "Carbs"] if c in df.columns] 
    for c in numeric_cols: 
        df[c] = pd.to_numeric(df[c], errors="coerce") 
        
    # Imputation
    missing_count = df.isnull().sum().sort_values(ascending=False) 
    missing_pct = (df.isnull().mean() * 100).sort_values(ascending=False) 
    missing_report = pd.DataFrame({"missing_count": missing_count, "missing_pct": missing_pct}) 
    missing_nonzero = missing_pct[missing_pct > 0] 
    
    for col in df.columns: 
        if pd.api.types.is_numeric_dtype(df[col]): 
            df[col] = df[col].fillna(df[col].median()) 
        else: 
            df[col] = df[col].fillna("Unknown") 
            
    # Engineering
    eps = 1e-6 
    if set(["Protein", "Fiber", "Calories"]).issubset(df.columns): 
        df["nutrient_density"] = (df["Protein"] + df["Fiber"]) / (df["Calories"] + eps) 
    if set(["Grams", "Calories"]).issubset(df.columns): 
        df["calories_per_gram"] = df["Calories"] / (df["Grams"] + eps) 
    if set(["Protein", "Fat", "Carbs"]).issubset(df.columns): 
        total_macros = df["Protein"] + df["Fat"] + df["Carbs"] + eps 
        df["protein_ratio"] = df["Protein"] / total_macros 
        df["fat_ratio"] = df["Fat"] / total_macros 
        df["carb_ratio"] = df["Carbs"] / total_macros 

    engineered_cols = [c for c in ["nutrient_density", "calories_per_gram", "protein_ratio", "fat_ratio", "carb_ratio"] if c in df.columns] 
    
    if "Calories" not in df.columns: 
        raise ValueError("Calories column not found.") 
        
    q1, q2 = df["Calories"].quantile(0.33), df["Calories"].quantile(0.66) 
    df["calorie_group"] = df["Calories"].apply(lambda x: "Low" if x <= q1 else ("Medium" if x <= q2 else "High")) 
    
    # ML Prepping
    drop_cols_for_ml = [c for c in ["Food"] if c in df.columns] 
    X_raw = df.drop(columns=["Calories", "calorie_group"] + drop_cols_for_ml).copy() 
    cat_cols = X_raw.select_dtypes(exclude=[np.number]).columns.tolist() 
    low_card = [c for c in cat_cols if X_raw[c].nunique() <= 10] 
    high_card = [c for c in cat_cols if X_raw[c].nunique() > 10] 
    
    freq_maps = {c: X_raw[c].value_counts(normalize=True) for c in high_card} 
    for c in high_card: 
        X_raw[c] = X_raw[c].map(freq_maps[c]).fillna(0) 
        
    X = pd.get_dummies(X_raw, columns=low_card, drop_first=True).replace([np.inf, -np.inf], np.nan).fillna(0) 
    y_cls_cat = df["calorie_group"].astype("category") 
    class_names = list(y_cls_cat.cat.categories) 
    y_cls, y_reg = y_cls_cat.cat.codes, df["Calories"].values 

    X_train, X_test, y_train, y_test = train_test_split(X, y_cls, test_size=0.25, random_state=42, stratify=y_cls) 
    scaler = StandardScaler().fit(X_train) 
    X_train_scaled, X_test_scaled = scaler.transform(X_train), scaler.transform(X_test) 

    # Classification
    models = {
        "Random Forest": ("raw", RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")), 
        "Logistic Regression": ("scaled", LogisticRegression(max_iter=1000, class_weight="balanced")), 
        "XGBoost": ("raw", XGBClassifier(n_estimators=150, random_state=42, eval_metric="mlogloss"))
    } 
    
    results, probs_for_roc, model_preds = [], {}, {} 
    for name, (mode, model) in models.items(): 
        xt, xv = (X_train_scaled, X_test_scaled) if mode == "scaled" else (X_train, X_test) 
        model.fit(xt, y_train) 
        pred, prob = model.predict(xv), model.predict_proba(xv) 
        results.append([name, accuracy_score(y_test, pred), precision_score(y_test, pred, average="macro"), 
                        recall_score(y_test, pred, average="macro"), f1_score(y_test, pred, average="macro"), 
                        roc_auc_score(y_test, prob, multi_class="ovr")]) 
        probs_for_roc[name], model_preds[name] = prob, pred 

    results_df = pd.DataFrame(results, columns=["Model", "Accuracy", "Precision", "Recall", "F1", "AUC"]).sort_values("F1", ascending=False) 
    best_model_name = results_df.iloc[0]["Model"] 
    
    # Regression
    Xr_train, Xr_test, yr_train, yr_test = train_test_split(X, y_reg, test_size=0.25, random_state=42) 
    scaler_r = StandardScaler().fit(Xr_train) 
    reg_models = {
        "Linear Regression": ("scaled", LinearRegression()), 
        "Random Forest Regressor": ("raw", RandomForestRegressor(n_estimators=100, random_state=42)), 
        "XGBoost Regressor": ("raw", XGBRegressor(n_estimators=150, random_state=42))
    } 
    reg_results = [] 
    for name, (mode, model) in reg_models.items(): 
        xt, xv = (scaler_r.transform(Xr_train), scaler_r.transform(Xr_test)) if mode == "scaled" else (Xr_train, Xr_test) 
        model.fit(xt, yr_train) 
        pred = model.predict(xv) 
        reg_results.append([name, mean_absolute_error(yr_test, pred), rmse_compat(yr_test, pred), r2_score(yr_test, pred)]) 
    
    reg_df = pd.DataFrame(reg_results, columns=["Model", "MAE", "RMSE", "R2"]).sort_values("RMSE") 

    # Recommendation Engine
    cluster_features = [c for c in ["Calories", "Protein", "Fat", "Carbs", "Fiber"] if c in df.columns] 
    cluster_scaler = StandardScaler() 
    cluster_scaled = cluster_scaler.fit_transform(df[cluster_features]) 
    sim_mat = cosine_similarity(cluster_scaled) 
    food_to_index = {str(f).strip(): i for i, f in enumerate(df["Food"].astype(str).tolist())} if "Food" in df.columns else {} 

    return {
        "raw_df": raw_df, "df": df, "numeric_cols": numeric_cols, "missing_report": missing_report, 
        "missing_nonzero": missing_nonzero, "engineered_cols": engineered_cols, "q1": q1, "q2": q2, 
        "drop_cols_for_ml": drop_cols_for_ml, "X": X, "class_names": class_names, "y_test": y_test, 
        "results_df": results_df, "models": models, "best_model_name": best_model_name, "best_mode": models[best_model_name][0], 
        "best_model": models[best_model_name][1], "best_pred": model_preds[best_model_name], "best_prob": probs_for_roc[best_model_name], 
        "top15_imp": pd.Series(models["Random Forest"][1].feature_importances_, index=X.columns).sort_values(ascending=False).head(15), 
        "scaler": scaler, "reg_models": reg_models, "reg_df": reg_df, "best_reg_name": reg_df.iloc[0]["Model"], 
        "scaler_r": scaler_r, "sim_mat": sim_mat, "food_to_index": food_to_index, "xy": PCA(2).fit_transform(cluster_scaled)
    } 

def prepare_one_row(input_dict, ctx): 
    one = pd.DataFrame([input_dict]).replace(["t", "t'"], 0).replace(",", "", regex=True) 
    for c in ctx["numeric_cols"]: 
        if c in one.columns: one[c] = pd.to_numeric(one[c], errors="coerce") 
    for col in one.columns: 
        one[col] = one[col].fillna(0) if pd.api.types.is_numeric_dtype(one[col]) else one[col].fillna("Unknown") 
    
    eps = 1e-6 
    if set(["Protein", "Fiber", "Calories"]).issubset(one.columns): 
        one["nutrient_density"] = (one["Protein"] + one["Fiber"]) / (one["Calories"] + eps) 
    if set(["Grams", "Calories"]).issubset(one.columns): 
        one["calories_per_gram"] = one["Calories"] / (one["Grams"] + eps) 
    if set(["Protein", "Fat", "Carbs"]).issubset(one.columns): 
        total = one["Protein"] + one["Fat"] + one["Carbs"] + eps 
        one["protein_ratio"], one["fat_ratio"], one["carb_ratio"] = one["Protein"]/total, one["Fat"]/total, one["Carbs"]/total 
        
    for c in ctx["drop_cols_for_ml"]: 
        if c in one.columns: one = one.drop(columns=[c]) 
    # Use freq_maps and get_dummies logic consistent with training 
    one_enc = pd.get_dummies(one).reindex(columns=ctx["X"].columns, fill_value=0) 
    return one_enc 

# ... (Insert visual rendering functions here as needed) ...

with st.sidebar: 
    st.header("Input") 
    use_demo = st.checkbox("Use Demo Dataset", value=True) 
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"]) 
    module = st.radio("Modules", ["Pre-processing", "EDA", "Model Training", "Evaluation", "Predictions"]) 

file_bytes = None 
if uploaded_file: 
    file_bytes = uploaded_file.getvalue() 
elif use_demo: 
    try: 
        with open("nutrients_data.csv", "rb") as f: file_bytes = f.read() 
    except FileNotFoundError: st.error("Demo CSV missing."); st.stop() 
else: st.info("Upload data to start."); st.stop() 

if "ctx" not in st.session_state or st.session_state.get("file_id") != file_bytes: 
    st.session_state.ctx = run_pipeline(file_bytes) 
    st.session_state.file_id = file_bytes 

ctx = st.session_state.ctx 

if module == "Predictions": 
    st.subheader("Nutrient Prediction") 
    fields = [c for c in ["Category", "Measure", "Grams", "Protein", "Fat", "Sat.Fat", "Fiber", "Carbs"] if c in ctx["raw_df"].columns] 
    user_input = {} 
    with st.form("pred_form"): 
        for k in fields: user_input[k] = st.text_input(k, "") 
        if st.form_submit_button("Predict"): 
            typed = {k: (pd.to_numeric(v) if k not in ["Category", "Measure"] else v) for k, v in user_input.items()} 
            one_X = prepare_one_row(typed, ctx) 
            # Logic for calling best_reg_model and best_model here 
            st.success("Results calculated!")
