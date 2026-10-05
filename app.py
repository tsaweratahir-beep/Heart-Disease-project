import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

st.set_page_config(page_title="Heart Disease Mining App", layout="wide")
st.title("❤️ Heart Disease Data Mining App")
st.caption("Domain: Healthcare | Dataset: Statlog Cleveland + Hungary")

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_raw():
    df = pd.read_csv("heart_statlog_cleveland_hungary_final.csv")
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]
    return df

df_raw = load_raw()

# ---------------- PREPROCESS ----------------
@st.cache_data
def preprocess(df):
    df = df.copy()

    # Step 1: Disguised missing values (0 -> NaN)
    df["cholesterol"] = df["cholesterol"].replace(0, np.nan)
    df["resting_bp_s"] = df["resting_bp_s"].replace(0, np.nan)

    # Step 2: Fill NaN with median
    df["cholesterol"] = df["cholesterol"].fillna(df["cholesterol"].median())
    df["resting_bp_s"] = df["resting_bp_s"].fillna(df["resting_bp_s"].median())

    # Step 3: Drop duplicates
    df = df.drop_duplicates().reset_index(drop=True)

    # Step 4: Outlier capping (IQR)
    num_cols = ["age", "resting_bp_s", "cholesterol", "max_heart_rate", "oldpeak"]
    for c in num_cols:
        Q1 = df[c].quantile(0.25)
        Q3 = df[c].quantile(0.75)
        IQR = Q3 - Q1
        low, high = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        df[c] = np.clip(df[c], low, high)

    # Step 5: Double-check no NaN remains
    df = df.dropna().reset_index(drop=True)

    # Step 6: Scaling
    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled[num_cols] = scaler.fit_transform(df[num_cols])

    return df, df_scaled

df, df_scaled = preprocess(df_raw)
num_cols = ["age", "resting_bp_s", "cholesterol", "max_heart_rate", "oldpeak"]

# Final safety check
if df_scaled.isnull().sum().sum() > 0:
    st.error("⚠️ Still NaN values in data:")
    st.write(df_scaled.isnull().sum())
    st.stop()

# ---------------- SIDEBAR ----------------
st.sidebar.title("⚙️ Menu")
page = st.sidebar.radio("Select operation:", [
    "📄 View Dataset",
    "🧹 Preprocessing",
    "📊 EDA",
    "⛏️ Clustering",
    "🎯 Prediction"
])

# ---------------- 1. VIEW ----------------
if page == "📄 View Dataset":
    st.subheader("Raw Dataset")
    st.dataframe(df_raw.head(20))
    st.write("Shape:", df_raw.shape)

# ---------------- 2. PREPROCESSING ----------------
elif page == "🧹 Preprocessing":
    st.subheader("Preprocessed & Scaled Data")
    st.dataframe(df_scaled.head(20))
    st.success("✔ Zeros fixed  ✔ Duplicates removed  ✔ Outliers capped  ✔ Scaled")

# ---------------- 3. EDA ----------------
elif page == "📊 EDA":
    st.subheader("Exploratory Data Analysis")
    choice = st.selectbox("Select plot:",
                          ["Histogram", "Boxplot", "Scatterplot", "Heatmap"])

    if choice == "Histogram":
        col = st.selectbox("Column:", num_cols)
        fig, ax = plt.subplots()
        sns.histplot(df[col], kde=True, ax=ax, color="teal")
        st.pyplot(fig)

    elif choice == "Boxplot":
        col = st.selectbox("Column:", num_cols)
        fig, ax = plt.subplots()
        sns.boxplot(x=df[col], ax=ax, color="orange")
        st.pyplot(fig)

    elif choice == "Scatterplot":
        x = st.selectbox("X:", num_cols, index=0)
        y = st.selectbox("Y:", num_cols, index=3)
        fig, ax = plt.subplots()
        sns.scatterplot(data=df, x=x, y=y, hue="target", ax=ax)
        st.pyplot(fig)

    elif choice == "Heatmap":
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
        st.pyplot(fig)

# ---------------- 4. CLUSTERING ----------------
elif page == "⛏️ Clustering":
    st.subheader("K-Means Clustering")
    k = st.slider("Choose k:", 2, 6, 3)

    X_cluster = df_scaled.drop("target", axis=1)

    # ---------- Extra safety ----------
    X_cluster = X_cluster.fillna(X_cluster.median())
    # ----------------------------------

    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    df["Cluster"] = km.fit_predict(X_cluster)

    st.write(df["Cluster"].value_counts())

    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x="age", y="cholesterol",
                    hue="Cluster", palette="viridis", ax=ax)
    st.pyplot(fig)

# ---------------- 5. PREDICTION ----------------
elif page == "🎯 Prediction":
    st.subheader("Heart Disease Prediction")
    model_name = st.selectbox("Choose model:",
                              ["Logistic Regression", "Random Forest"])

    X = df.drop(["target", "Cluster"], axis=1, errors="ignore")
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    model = (LogisticRegression(max_iter=2000)
             if model_name == "Logistic Regression"
             else RandomForestClassifier(n_estimators=150, random_state=42))
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    st.write(f"**Accuracy:** {accuracy_score(y_test, preds):.2%}")
    st.write("Confusion Matrix:", confusion_matrix(y_test, preds))

    st.markdown("### 🔎 Enter Patient Details")
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", 20, 100, 54)
        sex = st.selectbox("Sex (1=M, 0=F)", [1, 0])
        cp = st.selectbox("Chest Pain Type (1–4)", [1, 2, 3, 4])
        bp = st.number_input("Resting BP", 80, 200, 130)
        chol = st.number_input("Cholesterol", 100, 600, 240)
        fbs = st.selectbox("Fasting Blood Sugar>120", [0, 1])
    with col2:
        ecg = st.selectbox("Resting ECG (0–2)", [0, 1, 2])
        mhr = st.number_input("Max Heart Rate", 60, 220, 150)
        angina = st.selectbox("Exercise Angina", [0, 1])
        oldpeak = st.number_input("Oldpeak", -3.0, 7.0, 1.0)
        slope = st.selectbox("ST Slope (1–3)", [1, 2, 3])

    input_arr = np.array([[age, sex, cp, bp, chol, fbs,
                           ecg, mhr, angina, oldpeak, slope]])

    if st.button("🔮 Predict"):
        result = model.predict(input_arr)[0]
        if result == 1:
            st.error("⚠️ High risk of Heart Disease")
        else:
            st.success("✅ Low risk of Heart Disease")