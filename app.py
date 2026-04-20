# ============================================================
# Food Nutrition: Analysis and Visualization
# Streamlit Version (Optimized for Speed + Fuzzy Match)
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
/* ── Page background ─────────────────────────────────────── */
[data-testid="stAppViewContainer"] { background-color: #f5f7fa; }
[data-testid="stHeader"] { background-color: #f5f7fa; border-bottom: 1px solid #dde3ec; }
/* ── Sidebar ─────────────────────────────────────────────── */
[data-testid="stSidebar"] { background-color: #eef1f7; border-right: 1px solid #d4daea; }
[data-testid="stSidebar"] .stRadio label { color: #2d3a4a; font-weight: 500; }
[data-testid="stSidebar"] hr { border-color: #c8d0df; }
/* ── Main title & caption ────────────────────────────────── */
h1 { color: #1a2a3a !important; letter-spacing: -0.5px; border-bottom: 2px solid #c5d0e0; padding-bottom: 0.4rem; margin-bottom: 0.5rem; }
[data-testid="stCaptionContainer"] p { color: #5a6a7e !important; font-size: 0.88rem; }
/* ── Section subheaders ──────────────────────────────────── */
h2, h3 { color: #243447 !important; border-left: 4px solid #6b93c4; padding-left: 10px; margin-top: 1.4rem !important; }
h4 { color: #344a60 !important; }
/* ── Metric cards ────────────────────────────────────────── */
[data-testid="stMetric"] { background: #ffffff; border: 1px solid #d0daea; border-radius: 10px; padding: 14px 18px; box-shadow: 0 1px 4px rgba(0,0,0,0.06); }
[data-testid="stMetricLabel"] { color: #5a6a7e !important; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.05em; }
[data-testid="stMetricValue"] { color: #1a2a3a !important; font-weight: 700; }
/* ── Dataframes / tables ─────────────────────────────────── */
[data-testid="stDataFrame"] { border: 1px solid #d4daea; border-radius: 8px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }
/* ── Info / success / warning / error banners ────────────── */
[data-testid="stAlert"] { border-radius: 8px; border-left-width: 4px; }
div[data-baseweb="notification"][kind="info"] { background-color: #e8f0fb !important; border-left-color: #4a7fc1 !important; color: #1e3a5f !important; }
div[data-baseweb="notification"][kind="success"] { background-color: #e6f4ed !important; border-left-color: #2e8b57 !important; color: #1a3d2b !important; }
div[data-baseweb="notification"][kind="warning"] { background-color: #fdf4e3 !important; border-left-color: #d4900a !important; color: #5a3a00 !important; }
div[data-baseweb="notification"][kind="error"] { background-color: #fde8e8 !important; border-left-color: #c0392b !important; color: #5a1010 !important; }
/* ── Buttons ─────────────────────────────────────────────── */
[data-testid="stFormSubmitButton"] button, .stButton > button { background-color: #3a6fa8 !important; color: #ffffff !important; border: none !important; border-radius: 7px !important; padding: 0.45rem 1.4rem !important; font-weight: 600 !important; letter-spacing: 0.02em; box-shadow: 0 2px 6px rgba(58,111,168,0.25) !important; transition: background-color 0.18s ease, box-shadow 0.18s ease; }
[data-testid="stFormSubmitButton"] button:hover, .stButton > button:hover { background-color: #2d5a8a !important; box-shadow: 0 3px 10px rgba(45,90,138,0.35) !important; }
/* ── Text inputs ─────────────────────────────────────────── */
[data-testid="stTextInput"] input { background-color: #ffffff !important; border: 1px solid #b8c8dc !important; border-radius: 6px !important; color: #1a2a3a !important; padding: 0.35rem 0.6rem; transition: border-color 0.15s; }
[data-testid="stTextInput"] input:focus { border-color: #4a7fc1 !important; box-shadow: 0 0 0 3px rgba(74,127,193,0.15) !important; }
/* ── File uploader ───────────────────────────────────────── */
[data-testid="stFileUploader"] { background-color: #ffffff !important; border: 2px dashed #9ab3d0 !important; border-radius: 10px !important; padding: 1rem; }
/* ── Plot containers ─────────────────────────────────────── */
[data-testid="stImage"], .stPlotlyChart, canvas { border-radius: 8px; box-shadow: 0 1px 5px rgba(0,0,0,0.07); }
/* ── Form container ──────────────────────────────────────── */
[data-testid="stForm"] { background-color: #ffffff; border: 1px solid #d0daea; border-radius: 12px; padding: 1.2rem 1.4rem; box-shadow: 0 2px 8px rgba(0,0,0,0.05); }
</style>
""", unsafe_allow_html=True)

st.title("Food Nutrition: Analysis and Visualization")
st.caption("Upload nutrients_data.csv • Pre-processing • EDA • Model Training • Evaluation and Visualizations • Predictions")

def rmse_compat(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

@st.cache_resource(show_spinner=False)
def run_pipeline(file_bytes):
    df = pd.read_csv(io.BytesIO(file_bytes))
    df.columns = [c.strip() for c in df.columns]
    raw_df = df.copy()

    df = df.replace("t", 0)
    df = df.replace("t'", 0)
    df = df.replace(",", "", regex=True)

    if "Fiber" in df.columns:
        df["Fiber"] = df["Fiber"].astype(str).str.replace("a", "", regex=False)

    numeric_cols = []
    for c in ["Grams", "Calories", "Protein", "Fat", "Sat.Fat", "Fiber", "Carbs"]:
        if c in df.columns:
            numeric_cols.append(c)

    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    missing_count = df.isnull().sum().sort_values(ascending=False)
    missing_pct = (df.isnull().mean() * 100).sort_values(ascending=False)
    missing_report = pd.DataFrame({"missing_count": missing_count, "missing_pct": missing_pct})
    missing_nonzero = missing_pct[missing_pct > 0]

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna("Unknown")

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
        raise ValueError("Calories column not found. Cannot build regression/classification targets.")

    q1 = df["Calories"].quantile(0.33)
    q2 = df["Calories"].quantile(0.66)

    def calorie_group(x):
        if x <= q1:
            return "Low"
        elif x <= q2:
            return "Medium"
        else:
            return "High"

    df["calorie_group"] = df["Calories"].apply(calorie_group)

    drop_cols_for_ml = []
    # MODIFICATION 1: Removed "Measure" from this list so models can use it
    for c in ["Food"]: 
        if c in df.columns:
            drop_cols_for_ml.append(c)

    X_raw = df.drop(columns=["Calories", "calorie_group"] + drop_cols_for_ml).copy()
    
    cat_cols = X_raw.select_dtypes(exclude=[np.number]).columns.tolist()

    low_card = []
    high_card = []
    for c in cat_cols:
        nuniq = X_raw[c].nunique(dropna=True)
        if nuniq <= 10:
            low_card.append(c)
        else:
            high_card.append(c)

    freq_maps = {}
    for c in high_card:
        freq = X_raw[c].value_counts(normalize=True)
        freq_maps[c] = freq
        X_raw[c] = X_raw[c].map(freq).fillna(0)

    X = pd.get_dummies(X_raw, columns=low_card, drop_first=True)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    y_cls_cat = df["calorie_group"].astype("category")
    class_names = list(y_cls_cat.cat.categories)
    y_cls = y_cls_cat.cat.codes
    y_reg = df["Calories"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y_cls, test_size=0.25, random_state=42, stratify=y_cls)

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ==========================================
    # OPTIMIZED CLASSIFICATION MODELS
    # ==========================================
    models = {}
    models["Random Forest"] = ("raw", RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight="balanced"))
    models["Logistic Regression"] = ("scaled", LogisticRegression(max_iter=1000, class_weight="balanced"))
    models["XGBoost"] = ("raw", XGBClassifier(n_estimators=150, max_depth=5, learning_rate=0.1, subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0, random_state=42, eval_metric="mlogloss"))

    results = []
    probs_for_roc = {}
    model_preds = {}

    for name, (mode, model) in models.items():
        if mode == "scaled":
            model.fit(X_train_scaled, y_train)
            pred = model.predict(X_test_scaled)
            prob = model.predict_proba(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            prob = model.predict_proba(X_test)

        acc = accuracy_score(y_test, pred)
        prec = precision_score(y_test, pred, average="macro", zero_division=0)
        rec = recall_score(y_test, pred, average="macro", zero_division=0)
        f1 = f1_score(y_test, pred, average="macro", zero_division=0)
        try:
            auc = roc_auc_score(y_test, prob, multi_class="ovr")
        except Exception:
            auc = np.nan

        results.append([name, acc, prec, rec, f1, auc])
        probs_for_roc[name] = prob
        model_preds[name] = pred

    results_df = pd.DataFrame(results, columns=["Model", "Accuracy", "Precision(Macro)", "Recall(Macro)", "F1(Macro)", "ROC_AUC(OvR)"]).sort_values(by="F1(Macro)", ascending=False).reset_index(drop=True)

    # Get best classifier without retraining
    best_model_name = results_df.loc[0, "Model"]
    best_mode, best_model = models[best_model_name]
    best_pred = model_preds[best_model_name]
    best_prob = probs_for_roc[best_model_name]

    # Feature importances
    rf_for_imp = models["Random Forest"][1]
    imp = pd.Series(rf_for_imp.feature_importances_, index=X.columns).sort_values(ascending=False)
    top15_imp = imp.head(15)

    Xr_train, Xr_test, yr_train, yr_test = train_test_split(X, y_reg, test_size=0.25, random_state=42)

    scaler_r = StandardScaler()
    scaler_r.fit(Xr_train)
    Xr_train_scaled = scaler_r.transform(Xr_train)
    Xr_test_scaled = scaler_r.transform(Xr_test)

    # ==========================================
    # OPTIMIZED REGRESSION MODELS
    # ==========================================
    reg_models = {
        "Linear Regression": ("scaled", LinearRegression()),
        "Random Forest Regressor": ("raw", RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)),
        "XGBoost Regressor": ("raw", XGBRegressor(n_estimators=150, max_depth=5, learning_rate=0.1, subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0, random_state=42)),
    }

    reg_results = []
    reg_predictions = {}
    best_reg_name = None
    best_reg_rmse = None
    for name, (mode, model) in reg_models.items():
        if mode == "scaled":
            model.fit(Xr_train_scaled, yr_train)
            pred = model.predict(Xr_test_scaled)
        else:
            model.fit(Xr_train, yr_train)
            pred = model.predict(Xr_test)

        mae = mean_absolute_error(yr_test, pred)
        rmse = rmse_compat(yr_test, pred)
        r2 = r2_score(yr_test, pred)
        reg_results.append([name, mae, rmse, r2])
        reg_predictions[name] = pred

        if best_reg_rmse is None or rmse < best_reg_rmse:
            best_reg_rmse = rmse
            best_reg_name = name

    reg_df = pd.DataFrame(reg_results, columns=["Model", "MAE", "RMSE", "R2"]).sort_values("RMSE").reset_index(drop=True)

    cluster_features = [c for c in ["Calories", "Protein", "Fat", "Carbs", "Fiber"] if c in df.columns]
    cluster_data = df[cluster_features].copy()
    cluster_scaler = StandardScaler()
    cluster_scaled = cluster_scaler.fit_transform(cluster_data)
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(cluster_scaled)
    df["cluster"] = clusters

    pca = PCA(n_components=2, random_state=42)
    xy = pca.fit_transform(cluster_scaled)

    sim_features = cluster_features.copy()
    sim_mat = cosine_similarity(cluster_scaler.transform(df[sim_features]))
    food_to_index = {}
    if "Food" in df.columns:
        food_to_index = {str(f).strip(): i for i, f in enumerate(df["Food"].astype(str).tolist())}

    healthiest = pd.DataFrame()
    unhealthiest = pd.DataFrame()
    if "nutrient_density" in df.columns:
        tmp = df.copy()
        tmp["health_score"] = tmp["nutrient_density"] - (tmp["Calories"] / (tmp["Calories"].max() + 1e-6))
        healthiest = tmp.sort_values("health_score", ascending=False).head(10)
        unhealthiest = tmp.sort_values("health_score", ascending=True).head(10)

    top_calorie = pd.DataFrame()
    top_fat = pd.DataFrame()
    if "Food" in df.columns and "Calories" in df.columns:
        top_calorie = df.sort_values("Calories", ascending=False)[["Food", "Category", "Calories"]].head(10)
    if "Food" in df.columns and "Fat" in df.columns:
        top_fat = df.sort_values("Fat", ascending=False)[["Food", "Category", "Fat"]].head(10)

    return {
        "raw_df": raw_df, "df": df, "numeric_cols": numeric_cols, "missing_report": missing_report,
        "missing_nonzero": missing_nonzero, "engineered_cols": engineered_cols, "q1": q1, "q2": q2,
        "drop_cols_for_ml": drop_cols_for_ml, "X": X, "class_names": class_names, "y_test": y_test,
        "results_df": results_df, "models": models, "best_model_name": best_model_name,
        "best_mode": best_mode, "best_model": best_model, "best_pred": best_pred, "best_prob": best_prob,
        "top15_imp": top15_imp, "scaler": scaler, "X_train": X_train, "X_train_scaled": X_train_scaled,
        "y_train": y_train, "freq_maps": freq_maps, "low_card": low_card, "high_card": high_card,
        "reg_models": reg_models, "reg_df": reg_df, "best_reg_name": best_reg_name, "scaler_r": scaler_r,
        "Xr_train": Xr_train, "Xr_train_scaled": Xr_train_scaled, "yr_train": yr_train, "yr_test": yr_test,
        "reg_predictions": reg_predictions, "cluster_features": cluster_features, "xy": xy,
        "sim_mat": sim_mat, "food_to_index": food_to_index, "top_calorie": top_calorie, "top_fat": top_fat,
        "healthiest": healthiest, "unhealthiest": unhealthiest
    }

def prepare_one_row(input_dict, ctx):
    one = pd.DataFrame([input_dict]).copy()
    one = one.replace("t", 0).replace("t'", 0)
    one = one.replace(",", "", regex=True)

    for c in ctx["numeric_cols"]:
        if c in one.columns:
            one[c] = pd.to_numeric(one[c], errors="coerce")

    for col in one.columns:
        if pd.api.types.is_numeric_dtype(one[col]):
            one[col] = one[col].fillna(0)
        else:
            one[col] = one[col].fillna("Unknown")

    eps = 1e-6
    if set(["Protein", "Fiber", "Calories"]).issubset(one.columns):
        one["nutrient_density"] = (one["Protein"] + one["Fiber"]) / (one["Calories"] + eps)
    if set(["Grams", "Calories"]).issubset(one.columns):
        one["calories_per_gram"] = one["Calories"] / (one["Grams"] + eps)
    if set(["Protein", "Fat", "Carbs"]).issubset(one.columns):
        total_macros = one["Protein"] + one["Fat"] + one["Carbs"] + eps
        one["protein_ratio"] = one["Protein"] / total_macros
        one["fat_ratio"] = one["Fat"] / total_macros
        one["carb_ratio"] = one["Carbs"] / total_macros

    for c in ctx["drop_cols_for_ml"]:
        if c in one.columns:
            one = one.drop(columns=[c])

    for c, freq in ctx["freq_maps"].items():
        if c in one.columns:
            one[c] = one[c].map(freq).fillna(0)

    one_enc = pd.get_dummies(one, columns=[c for c in ctx["low_card"] if c in one.columns], drop_first=True)

    for col in ctx["X"].columns:
        if col not in one_enc.columns:
            one_enc[col] = 0
    one_enc = one_enc[ctx["X"].columns]
    one_enc = one_enc.replace([np.inf, -np.inf], np.nan).fillna(0)
    return one_enc

def recommend_healthier(df, sim_mat, food_to_index, food_name, top_n=5):
    # 1. Try an exact match first
    if food_name in food_to_index:
        target_food = food_name
    else:
        # 2. If no exact match, try a partial (case-insensitive) match
        matches = df[df['Food'].astype(str).str.contains(food_name, case=False, na=False)]['Food'].tolist()
        if not matches:
            return pd.DataFrame(), None # Nothing found at all
        
        # Grab the first partial match we found
        target_food = matches[0]

    # Now proceed with the math using the target_food
    i = food_to_index[target_food]
    sims = sim_mat[i].copy()
    sims[i] = -1
    cand_idx = np.argsort(sims)[::-1][:50]
    base_cal = df.loc[i, "Calories"]
    recs = df.loc[cand_idx, ["Food", "Category", "Calories"]].copy()
    recs["similarity"] = sims[cand_idx]

    if "nutrient_density" in df.columns:
        recs["nutrient_density"] = df.loc[cand_idx, "nutrient_density"].values
        recs = recs[recs["Calories"] < base_cal]
        recs = recs.sort_values(["nutrient_density", "similarity"], ascending=[False, False]).head(top_n)
    else:
        recs = recs[recs["Calories"] < base_cal].sort_values("similarity", ascending=False).head(top_n)
        
    return recs, target_food

def render_data_info(df):
    st.subheader("Data Information")
    c1, c2 = st.columns(2)
    c1.metric("Rows", int(df.shape[0]))
    c2.metric("Columns", int(df.shape[1]))
    st.write("First 5 rows:")
    st.dataframe(df.head(), use_container_width=True)
    st.write("Column names:")
    st.write(list(df.columns))

def render_preprocessing(ctx):
    st.subheader("Pre-processing")
    st.dataframe(ctx["missing_report"], use_container_width=True)
    if not ctx["missing_nonzero"].empty:
        fig = plt.figure(figsize=(10, 4))
        ctx["missing_nonzero"].plot(kind="bar")
        plt.title("Missing Value % by Column")
        plt.ylabel("Missing %")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("No missing values found.")
    st.write("Engineered columns added:", ctx["engineered_cols"])
    st.write("Calorie thresholds:", {"q1 (33%)": float(ctx["q1"]), "q2 (66%)": float(ctx["q2"])})

def render_eda(ctx):
    st.subheader("EDA")
    df = ctx["df"]
    for col in [c for c in ["Calories", "Protein", "Fat", "Carbs"] if c in df.columns]:
        fig = plt.figure(figsize=(7, 4))
        sns.histplot(df[col], kde=True, bins=30)
        plt.title(f"Distribution of {col}")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    if "Category" in df.columns and set(["Protein", "Fat", "Carbs"]).issubset(df.columns):
        cat_means = df.groupby("Category")[["Protein", "Fat", "Carbs"]].mean().sort_values("Protein", ascending=False).head(12)
        fig = plt.figure(figsize=(12, 4))
        cat_means.plot(kind="bar", figsize=(12, 4))
        plt.title("Average Macro-Nutrients by Category (Top 12 categories)")
        plt.ylabel("Average (grams)")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    fig = plt.figure(figsize=(8, 6))
    corr = ctx["df"][ctx["numeric_cols"]].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Correlation Heatmap (Numeric Features)")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    for col in [c for c in ["Calories", "Fat"] if c in df.columns]:
        fig = plt.figure(figsize=(7, 3))
        sns.boxplot(x=df[col])
        plt.title(f"Outlier Check: {col} (Boxplot)")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    if not ctx["top_calorie"].empty:
        st.write("Top 10 highest-calorie foods:")
        st.dataframe(ctx["top_calorie"], use_container_width=True)
    if not ctx["top_fat"].empty:
        st.write("Top 10 highest-fat foods:")
        st.dataframe(ctx["top_fat"], use_container_width=True)

    if "Protein" in df.columns and "Calories" in df.columns:
        fig = plt.figure(figsize=(7, 4))
        sns.scatterplot(data=df, x="Calories", y="Protein", hue="Category" if "Category" in df.columns else None, legend=False)
        plt.title("Protein vs Calories")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    if "Fat" in df.columns and "Carbs" in df.columns:
        fig = plt.figure(figsize=(7, 4))
        sns.scatterplot(data=df, x="Carbs", y="Fat", hue="Category" if "Category" in df.columns else None, legend=False)
        plt.title("Fat vs Carbs")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

def render_model_training(ctx):
    st.subheader("Model Training")
    st.markdown("#### Classification Model Comparison")
    st.dataframe(ctx["results_df"], use_container_width=True)
    st.write("Best classifier:", ctx["best_model_name"])
    st.markdown("#### Regression Model Comparison")
    st.dataframe(ctx["reg_df"], use_container_width=True)
    st.write("Best regressor:", ctx["best_reg_name"])
    fig = plt.figure(figsize=(5, 3))
    sns.countplot(x="calorie_group", data=ctx["df"], order=["Low", "Medium", "High"])
    plt.title("Calorie Group Distribution (Low/Medium/High)")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

def render_eval_visualizations(ctx):
    st.subheader("Evaluation and Visualizations")
    st.write("Confusion Matrix (Best Classifier):")
    st.dataframe(pd.DataFrame(confusion_matrix(ctx["y_test"], ctx["best_pred"])), use_container_width=True)
    st.code(classification_report(ctx["y_test"], ctx["best_pred"], target_names=ctx["class_names"]))

    y_test_bin = label_binarize(ctx["y_test"], classes=list(range(len(ctx["class_names"]))))
    fig = plt.figure(figsize=(8, 6))
    for i, cls in enumerate(ctx["class_names"]):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], ctx["best_prob"][:, i])
        auc_i = roc_auc_score(y_test_bin[:, i], ctx["best_prob"][:, i])
        plt.plot(fpr, tpr, label=f"{cls} (AUC={auc_i:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.title(f"ROC Curves (Best Classifier: {ctx['best_model_name']}, OvR)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.write("Top 15 important features (Random Forest):")
    st.dataframe(ctx["top15_imp"].to_frame("importance"), use_container_width=True)
    fig = plt.figure(figsize=(9, 5))
    sns.barplot(x=ctx["top15_imp"].values, y=ctx["top15_imp"].index)
    plt.title("Top 15 Feature Importances (Random Forest)")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    fig = plt.figure(figsize=(7, 5))
    sns.scatterplot(x=ctx["xy"][:, 0], y=ctx["xy"][:, 1], hue=ctx["df"]["cluster"], palette="tab10", legend="full")
    plt.title("Food Clusters (PCA view)")
    plt.xlabel("PCA-1")
    plt.ylabel("PCA-2")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    if not ctx["healthiest"].empty and not ctx["unhealthiest"].empty:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].barh(ctx["healthiest"]["Food"].astype(str), ctx["healthiest"]["health_score"])
        axes[0].set_title("Top 10 Healthiest (heuristic score)")
        axes[0].invert_yaxis()
        axes[1].barh(ctx["unhealthiest"]["Food"].astype(str), ctx["unhealthiest"]["health_score"])
        axes[1].set_title("Top 10 Unhealthiest (heuristic score)")
        axes[1].invert_yaxis()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

def render_predictions(ctx):
    st.subheader("Predictions")
    st.write("Enter input values manually (no default values).")
    
    # MODIFICATION 2: Added "Measure" right after "Category"
    fields = [c for c in ["Category", "Measure", "Grams", "Protein", "Fat", "Sat.Fat", "Fiber", "Carbs"] if c in ctx["raw_df"].columns or c in ["Category", "Measure"]]
    
    user_input = {}
    with st.form("nutrition_prediction_form"):
        cols = st.columns(2)
        for i, k in enumerate(fields):
            with cols[i % 2]:
                user_input[k] = st.text_input(k, value="", key=f"user_{k}")
        rec_food = st.text_input("Food Name for Healthier Alternatives", value="", key="rec_food")
        submitted = st.form_submit_button("Predict")

    if not submitted:
        return

    missing = [k for k, v in user_input.items() if str(v).strip() == ""]
    if missing:
        st.error("Please enter all required values before prediction.")
        st.write("Missing fields:", missing)
        return

    typed = {}
    numeric_fields = {"Grams", "Protein", "Fat", "Sat.Fat", "Fiber", "Carbs", "Calories"}
    for k, v in user_input.items():
        vv = str(v).strip()
        if k in numeric_fields:
            num = pd.to_numeric(vv, errors="coerce")
            if pd.isna(num):
                st.error(f"Invalid numeric value for {k}.")
                return
            typed[k] = float(num)
        else:
            typed[k] = vv

    typed.setdefault("Calories", 0.0)
    one_X = prepare_one_row(typed, ctx)

    # NO RETRAINING: directly use the already-fitted model from memory
    best_reg_mode, best_reg_model = ctx["reg_models"][ctx["best_reg_name"]]
    if best_reg_mode == "scaled":
        cal_pred = float(best_reg_model.predict(ctx["scaler_r"].transform(one_X))[0])
    else:
        cal_pred = float(best_reg_model.predict(one_X)[0])

    typed["Calories"] = cal_pred
    one_X = prepare_one_row(typed, ctx)

    # NO RETRAINING: directly use the already-fitted model from memory
    if ctx["best_mode"] == "scaled":
        prob = ctx["best_model"].predict_proba(ctx["scaler"].transform(one_X))[0]
        pred_code = int(ctx["best_model"].predict(ctx["scaler"].transform(one_X))[0])
    else:
        prob = ctx["best_model"].predict_proba(one_X)[0]
        pred_code = int(ctx["best_model"].predict(one_X)[0])

    st.success(f"Predicted Calories: {cal_pred:.2f}")
    st.info(f"Predicted Calorie Group: {ctx['class_names'][pred_code]}")
    st.dataframe(pd.DataFrame({"Class": ctx["class_names"], "Probability": prob}), use_container_width=True)

    if rec_food.strip() and "Food" in ctx["df"].columns:
        recs, matched_food = recommend_healthier(ctx["df"], ctx["sim_mat"], ctx["food_to_index"], rec_food.strip(), top_n=5)
        
        if recs.empty:
            st.warning(f"No healthier alternatives found (or no food containing '{rec_food}' was found).")
        else:
            st.write(f"Healthier alternatives for: **{matched_food}**")
            st.dataframe(recs, use_container_width=True)

with st.sidebar:
    st.header("Input")
    uploaded_file = st.file_uploader("Upload nutrients_data.csv", type=["csv"], accept_multiple_files=False)
    st.markdown("---")
    module = st.radio("Modules", ["Pre-processing", "EDA", "Model Training", "Evaluation and Visualizations", "Predictions"])

# ==========================================
# THE MAGIC: st.session_state optimization
# ==========================================

if uploaded_file is None:
    st.info("Upload `nutrients_data.csv` from the left sidebar to continue.")
    st.stop()

# 1. Create a unique ID for the file to detect if the user uploads a new one
file_id = uploaded_file.name + str(uploaded_file.size)

# 2. If it's a new file, clear the old memory
if "file_id" not in st.session_state or st.session_state.file_id != file_id:
    st.session_state.file_id = file_id
    st.session_state.ctx = None 

# 3. Read the basic preview dataframe
try:
    file_bytes = uploaded_file.getvalue()
    preview_df = pd.read_csv(io.BytesIO(file_bytes))
except Exception as ex:
    st.error(f"Could not read dataset: {ex}")
    st.stop()

if module == "Pre-processing":
    render_data_info(preview_df)

# 4. Only run the heavy ML pipeline if it's not already in memory
if st.session_state.ctx is None:
    with st.spinner(" Crunching data & training XGBoost/RF Models... "):
        try:
            st.session_state.ctx = run_pipeline(file_bytes)
        except Exception as ex:
            st.error(f"Pipeline failed: {ex}")
            st.stop()

# 5. Instantly load the context from server RAM
ctx = st.session_state.ctx

# 6. Render the selected module
if module == "Pre-processing":
    render_preprocessing(ctx)
elif module == "EDA":
    render_eda(ctx)
elif module == "Model Training":
    render_model_training(ctx)
elif module == "Evaluation and Visualizations":
    render_eval_visualizations(ctx)
elif module == "Predictions":
    render_predictions(ctx)
