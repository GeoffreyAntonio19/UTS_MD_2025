import streamlit as st
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

# ---- Title ----
st.title('Hotel Booking Cancellation Prediction')
st.info('This application predicts whether a hotel booking will be cancelled based on various features.')

# ---- Load Data ----
@st.cache_data
def load_data():
    df = pd.read_csv('Dataset_B_hotel.csv')
    return df

df = load_data()

# ---- Data Exploration ----
with st.expander('**Dataset Preview**'):
    st.write('This is the raw dataset:')
    st.dataframe(df)

# ---- Preprocessing Function ----
def preprocess_data(df):
    df.drop(columns=["Booking_ID"], inplace=True)
    y = df["booking_status"] if "booking_status" in df.columns else None
    X = df.drop(columns=["booking_status"]) if y is not None else df.copy()

    # Identify categorical and numerical columns
    categorical_columns = X.select_dtypes(include=["object"]).columns.tolist()
    numerical_columns = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    # Imputers
    imputer_cat = SimpleImputer(strategy='most_frequent')
    imputer_num = SimpleImputer(strategy='mean')

    X[categorical_columns] = imputer_cat.fit_transform(X[categorical_columns])
    X[numerical_columns] = imputer_num.fit_transform(X[numerical_columns])

    # Label Encoding
    label_encoders = {}
    for col in categorical_columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le

    # Encode target
    if y is not None:
        le_target = LabelEncoder()
        y_encoded = le_target.fit_transform(y)
        label_encoders["booking_status"] = le_target
    else:
        y_encoded = None

    # Scaling
    scaler = StandardScaler()
    X[numerical_columns] = scaler.fit_transform(X[numerical_columns])

    # Combine again
    if y_encoded is not None:
        df_processed = X.copy()
        df_processed["booking_status"] = y_encoded
    else:
        df_processed = X

    return df_processed, label_encoders, imputer_cat, imputer_num, scaler, categorical_columns, numerical_columns

df, label_encoders, imputer_cat, imputer_num, scaler, cat_cols, num_cols = preprocess_data(df)

# ---- Train Model ----
X = df.drop(columns=["booking_status"])
y = df["booking_status"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = XGBClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ---- Input Features ----
st.subheader('**Input Features**')
user_data = pd.DataFrame({
    "no_of_adults": [st.slider("Number of Adults", 0, 10, 1)],
    "no_of_children": [st.slider("Number of Children", 0, 10, 0)],
    "no_of_weekend_nights": [st.slider("Weekend Nights", 0, 7, 1)],
    "no_of_week_nights": [st.slider("Week Nights", 0, 20, 2)],
    "type_of_meal_plan": [st.selectbox("Meal Plan", ("Meal Plan 1", "Meal Plan 2", "Meal Plan 3", "Not Selected"))],
    "required_car_parking_space": [st.selectbox("Car Parking Space", (0, 1))],
    "room_type_reserved": [st.selectbox("Room Type", ("Room_Type 1", "Room_Type 2", "Room_Type 3", "Room_Type 4", "Room_Type 5", "Room_Type 6", "Room_Type 7"))],
    "lead_time": [st.slider("Lead Time", 0, 450, 50)],
    "arrival_year": [st.selectbox("Arrival Year", (2017, 2018))],
    "arrival_month": [st.selectbox("Arrival Month", list(range(1, 13)))],
    "arrival_date": [st.slider("Arrival Date", 1, 31, 1)],
    "market_segment_type": [st.selectbox("Market Segment", ("Offline", "Online", "Corporate", "Aviation", "Complementary"))],
    "repeated_guest": [st.selectbox("Repeated Guest", (0, 1))],
    "no_of_previous_cancellations": [st.slider("Previous Cancellations", 0, 20, 0)],
    "no_of_previous_bookings_not_canceled": [st.slider("Previous Bookings Not Canceled", 0, 100, 0)],
    "avg_price_per_room": [st.slider("Avg Price/Room", 0.0, 1000.0, 100.0)],
    "no_of_special_requests": [st.slider("Special Requests", 0, 5, 1)],
})

# ---- Display Input ----
st.subheader("User Input (Original)")
st.dataframe(user_data, use_container_width=True)

# ---- Preprocess User Input ----
user_input = user_data.drop(columns=["Booking_ID"]).copy()

# Encode categorical columns
for col in cat_cols:
    if col in user_input.columns:
        if user_input[col][0] not in label_encoders[col].classes_:
            user_input[col] = label_encoders[col].transform([label_encoders[col].classes_[0]])
        else:
            user_input[col] = label_encoders[col].transform(user_input[col])

# Impute
user_input[cat_cols] = imputer_cat.transform(user_input[cat_cols])
user_input[num_cols] = imputer_num.transform(user_input[num_cols])

# Scale
user_input[num_cols] = scaler.transform(user_input[num_cols])

# ---- Prediction ----
if st.button("Predict Cancellation"):
    prediction = model.predict(user_input)
    proba = model.predict_proba(user_input)

    if "booking_status" in label_encoders:
        label_target = label_encoders["booking_status"].classes_
    else:
        label_target = ["Not Canceled", "Canceled"]

    pred_label = label_target[prediction[0]]

    st.subheader("Prediction Result")
    st.success(f"**Predicted Status: {pred_label}**")
    st.write("**Prediction Probabilities:**")
    st.dataframe(pd.DataFrame(proba, columns=label_target).style.format("{:.4f}"))
