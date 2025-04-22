import streamlit as st
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# ---- Title ----
st.title('Hotel Booking Cancellation Prediction')
st.info('This application predicts whether a hotel booking will be cancelled based on various features.')

# ---- Load Data ----
@st.cache_data
def load_data():
    df = pd.read_csv('Dataset_B_hotel.csv')
    return df

df = load_data()
df.drop(columns=["Booking_ID"])

# ---- Data Exploration ----
with st.expander('**Dataset Preview**'):
    st.write('This is the raw dataset:')
    st.dataframe(df)

# ---- Data Preprocessing ----
def preprocess_data(df):
    label_encoders = {}
    categorical_columns = [
        "type_of_meal_plan", "room_type_reserved", "market_segment_type", "booking_status"
    ]

    for col in categorical_columns:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le

    return df, label_encoders

df, label_encoders = preprocess_data(df)

# ---- Train Model ----
X = df.drop(columns=["booking_status"])
y = df["booking_status"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = XGBClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ---- User Input Features ----
st.subheader('**Input Features**')
user_data = pd.DataFrame({
    "Booking_ID": [st.text_input("Booking ID")],
    "no_of_adults": [st.slider("Number of Adults", min_value=0, max_value=10, value=1)],
    "no_of_children": [st.slider("Number of Children", min_value=0, max_value=10, value=0)],
    "no_of_weekend_nights": [st.slider("Number of Weekend Nights", min_value=0, max_value=7, value=1)],
    "no_of_week_nights": [st.slider("Number of Week Nights", min_value=0, max_value=20, value=2)],
    "type_of_meal_plan": [st.selectbox("Meal Plan", ("Meal Plan 1", "Meal Plan 2", "Meal Plan 3", "Not Selected"))],
    "required_car_parking_space": [st.selectbox("Car Parking Space", (0, 1))],
    "room_type_reserved": [st.selectbox("Room Type Reserved", ("Room_Type 1", "Room_Type 2", "Room_Type 3", "Room_Type 4", "Room_Type 5", "Room_Type 6", "Room_Type 7"))],
    "lead_time": [st.slider("Lead Time", min_value=0, max_value=450, value=50)],
    "arrival_year": [st.selectbox("Arrival Year", (2017, 2018))],
    "arrival_month": [st.selectbox("Arrival Month", (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12))],
    "arrival_date": [st.slider("Arrival Date", min_value=1, max_value=31, value=1)],
    "market_segment_type": [st.selectbox("Market Segment", ("Offline", "Online", "Corporate", "Aviation", "Complementary"))],
    "repeated_guest": [st.selectbox("Repeated Guest", (0, 1))],
    "no_of_previous_cancellations": [st.slider("Previous Cancellations", min_value=0, max_value=20, value=0)],
    "no_of_previous_bookings_not_canceled": [st.slider("Previous Bookings Not Canceled", min_value=0, max_value=100, value=0)],
    "avg_price_per_room": [st.slider("Average Price per Room", min_value=0.0, max_value=1000.0, value=100.0)],
    "no_of_special_requests": [st.slider("Number of Special Requests", min_value=0, max_value=5, value=1)],
})

# ---- Display User Input (Original Data) ----
st.subheader("User Input Data")
user_data_original = user_data.copy() # Save the original input data before encoding
st.dataframe(user_data_original, use_container_width=True)

# ---- Encode User Input (One-by-One Handling) ----
user_data_encoded = user_data.copy()
for col in label_encoders:
    if col in user_data_encoded.columns:
        le = label_encoders[col]
        if user_data_encoded[col][0] not in le.classes_:
            user_data_encoded[col] = le.transform([le.classes_[0]])  # Use default (first class)
        else:
            user_data_encoded[col] = le.transform(user_data_encoded[col])

# ---- Make Prediction ----
if st.button("Predict Cancellation"):
    # Predict class and probabilities
    prediction_proba = model.predict_proba(user_data_encoded)
    prediction = model.predict(user_data_encoded)
    predicted_class = 'Cancelled' if prediction[0] == 1 else 'Not Cancelled'

    # Get class names from encoder
    class_names = label_encoders["booking_status"].classes_

    # Create dataframe for probabilities with class names as header
    df_proba = pd.DataFrame(prediction_proba, columns=class_names)

    # Display the prediction results and probability table
    st.subheader("Prediction Results")
    st.dataframe(df_proba.style.format("{:.4f}"), use_container_width=True)

    st.success(f"Predicted Cancellation Status: **{predicted_class}**")
