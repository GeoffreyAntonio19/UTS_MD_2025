import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from xgboost import XGBClassifier

# ---------------------- TITLE ----------------------
st.set_page_config(layout="wide")
st.title("Hotel Booking Cancellation Prediction")
st.write("Upload a CSV file to predict the booking status.")

# ---------------------- DATA UPLOAD ----------------------
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # ---------------------- DATA PREPROCESSING ----------------------
    def preprocess_data(df):
        # Impute missing values
        imputer = SimpleImputer(strategy='most_frequent')
        df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

        label_encoders = {}
        categorical_columns = [
            "type_of_meal_plan", "room_type_reserved", "market_segment_type"
        ]

        # Encode categorical features
        for col in categorical_columns:
            if col in df_imputed.columns:
                le = LabelEncoder()
                df_imputed[col] = le.fit_transform(df_imputed[col])
                label_encoders[col] = le

        # Encode target label
        le_target = LabelEncoder()
        df_imputed["booking_status"] = le_target.fit_transform(df_imputed["booking_status"])
        label_encoders["booking_status"] = le_target

        # Scale numerical features
        numerical_columns = df_imputed.select_dtypes(include=['int64', 'float64']).columns.drop("booking_status")
        scaler = StandardScaler()
        df_imputed[numerical_columns] = scaler.fit_transform(df_imputed[numerical_columns])

        return df_imputed, label_encoders, imputer, scaler

    df, label_encoders, imputer, scaler = preprocess_data(df)

    # ---------------------- MODEL TRAINING ----------------------
    X = df.drop(columns=["booking_status"])
    y = df["booking_status"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # ---------------------- METRIC EVALUATION ----------------------
    st.subheader("Model Evaluation Metrics")
    st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    st.write(f"Precision: {precision_score(y_test, y_pred):.2f}")
    st.write(f"Recall: {recall_score(y_test, y_pred):.2f}")
    st.write(f"F1 Score: {f1_score(y_test, y_pred):.2f}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_encoders["booking_status"].classes_, yticklabels=label_encoders["booking_status"].classes_)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    st.pyplot(fig)

    # ---------------------- PREDIKSI USER ----------------------
    st.subheader("Make a Booking Prediction")
    user_input = {}
    for col in X.columns:
        user_input[col] = st.number_input(f"{col}", value=0.0)

    if st.button("Predict"):
        input_df = pd.DataFrame([user_input])
        prediction = model.predict(input_df)
        predicted_class = label_encoders["booking_status"].inverse_transform([prediction[0]])[0]
        st.success(f"Predicted Booking Status: **{predicted_class}**")
