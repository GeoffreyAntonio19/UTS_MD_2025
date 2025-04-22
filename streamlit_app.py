# streamlit_app.py
import streamlit as st
import pandas as pd
import joblib

# Load model, preprocessor, dan label encoder
model = joblib.load("xgb_model.pkl")
preprocessor = joblib.load("preprocessor.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Fungsi prediksi
def predict_new_data(new_data: pd.DataFrame):
    transformed = preprocessor.transform(new_data)
    pred_encoded = model.predict(transformed)
    pred_decoded = label_encoder.inverse_transform(pred_encoded)
    return pred_decoded[0]

st.title("Hotel Booking Cancellation Prediction")

# Contoh data uji
if st.button("Test Case 1 (Canceled)"):
    test_data = pd.DataFrame([{
        'no_of_adults': 2,
        'no_of_children': 1,
        'no_of_weekend_nights': 3,
        'no_of_week_nights': 4,
        'type_of_meal_plan': 'Meal Plan 1',
        'required_car_parking_space': 0,
        'room_type_reserved': 'Room_Type 1',
        'lead_time': 300,
        'arrival_year': 2018,
        'arrival_month': 12,
        'arrival_date': 20,
        'market_segment_type': 'Online',
        'repeated_guest': 0,
        'no_of_previous_cancellations': 2,
        'no_of_previous_bookings_not_canceled': 0,
        'avg_price_per_room': 180.0,
        'no_of_special_requests': 0
    }])
    pred = predict_new_data(test_data)
    st.write("Prediction:", pred)

if st.button("Test Case 2 (Not Canceled)"):
    test_data = pd.DataFrame([{
        'no_of_adults': 1,
        'no_of_children': 0,
        'no_of_weekend_nights': 1,
        'no_of_week_nights': 2,
        'type_of_meal_plan': 'Meal Plan 1',
        'required_car_parking_space': 1,
        'room_type_reserved': 'Room_Type 1',
        'lead_time': 20,
        'arrival_year': 2018,
        'arrival_month': 7,
        'arrival_date': 10,
        'market_segment_type': 'Offline',
        'repeated_guest': 1,
        'no_of_previous_cancellations': 0,
        'no_of_previous_bookings_not_canceled': 3,
        'avg_price_per_room': 80.0,
        'no_of_special_requests': 2
    }])
    pred = predict_new_data(test_data)
    st.write("Prediction:", pred)
