import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

st.set_page_config(page_title="Churn Dashboard", layout="wide")

st.title("📊 Customer Churn Prediction")

@st.cache_data
def preprocess_data(df):
    df = df.copy()
    if 'customerID' in df.columns:
        df.drop('customerID', axis=1, inplace=True)
    if 'Churn' in df.columns:
        df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].fillna("Unknown")
        df[col] = LabelEncoder().fit_transform(df[col])
    df = df.fillna(0)
    return df

@st.cache_data
def load_model():
    return joblib.load("model.pkl")

menu = st.sidebar.radio("Menu", ["Upload & EDA", "Model Training", "Live Prediction"])

if menu == "Upload & EDA":
    uploaded = st.file_uploader("📁 Upload CSV", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        st.session_state.df = df
        st.write("📌 Dataset Preview:")
        st.dataframe(df.head())

        st.subheader("📈 Data Summary")
        st.text(f"Shape: {df.shape}")
        st.write("Missing Values:")
        st.dataframe(df.isnull().sum())

        st.subheader("🔢 Univariate Distribution")
        for col in df.select_dtypes(include=['int64', 'float64']):
            fig, ax = plt.subplots()
            sns.histplot(df[col], kde=True, ax=ax)
            st.pyplot(fig)

elif menu == "Model Training":
    if 'df' not in st.session_state:
        st.warning("Please upload dataset first from 'Upload & EDA'")
    else:
        df = preprocess_data(st.session_state.df)

        X = df.drop('Churn', axis=1)
        y = df['Churn']
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y)

        model_type = st.selectbox("Select Model", ["Logistic Regression", "Random Forest", "SVM"])
        if model_type == "Logistic Regression":
            model = LogisticRegression(max_iter=1000)
        elif model_type == "Random Forest":
            model = RandomForestClassifier()
        else:
            model = SVC(probability=True)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        st.success(f"✅ Accuracy: {acc:.2f}")
        st.text("Classification Report")
        st.text(classification_report(y_test, y_pred))

        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        st.pyplot(fig)

        joblib.dump(model, "model.pkl")
        st.success("✅ Model Saved as `model.pkl`")

elif menu == "Live Prediction":
    try:
        model = load_model()
        st.success("✅ Loaded model.pkl")
        scaler = StandardScaler()

        if 'df' in st.session_state:
            df = preprocess_data(st.session_state.df)
            X = df.drop('Churn', axis=1)
            scaler.fit(X)
            input_data = {}
            st.subheader("📝 Enter Input Features")
            for col in X.columns:
                input_data[col] = st.number_input(col, value=0.0)

            if st.button("🔮 Predict"):
                input_df = pd.DataFrame([input_data])
                input_scaled = scaler.transform(input_df)
                pred = model.predict(input_scaled)[0]
                prob = model.predict_proba(input_scaled)[0][1]
                if pred == 1:
                    st.error(f"❗ Predicted: YES (Churn Likely) - Confidence: {prob:.2%}")
                else:
                    st.success(f"✅ Predicted: NO (Customer Likely to Stay) - Confidence: {1 - prob:.2%}")
        else:
            st.warning("Please upload dataset first for consistent feature structure.")
    except:
        st.error("⚠️ `model.pkl` not found. Please train the model first.")
