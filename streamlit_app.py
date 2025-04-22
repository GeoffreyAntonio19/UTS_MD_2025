import streamlit as st
import pandas as pd
import pickle

# Load model, preprocessor, and label encoder
@st.cache_resource
def load_model():
    with open("xgb_model.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_preprocessor():
    with open("preprocessor.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_label_encoder():
    with open("label_encoder.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()
preprocessor = load_preprocessor()
label_encoder = load_label_encoder()

st.title("Hotel Booking Cancellation Prediction 🏨")

# User input form
no_of_adults = st.number_input("Number of Adults", min_value=0, max_value=10, value=2)
no_of_children = st.number_input("Number of Children", min_value=0, max_value=10, value=0)
no_of_weekend_nights = st.slider("Number of Weekend Nights", 0, 10, 1)
no_of_week_nights = st.slider("Number of Week Nights", 0, 20, 2)
type_of_meal_plan = st.selectbox("Meal Plan", ['Meal Plan 1', 'Not Selected', 'Meal Plan 2', 'Meal Plan 3'])
required_car_parking_space = st.selectbox("Requires Parking Space?", [0, 1])
room_type_reserved = st.selectbox("Reserved Room Type", [
    'Room_Type 1', 'Room_Type 2', 'Room_Type 3', 'Room_Type 4',
    'Room_Type 5', 'Room_Type 6', 'Room_Type 7'
])
lead_time = st.number_input("Lead Time (days before check-in)", min_value=0, max_value=500, value=30)
arrival_year = st.selectbox("Arrival Year", [2017, 2018])
arrival_month = st.selectbox("Arrival Month", list(range(1, 13)))
arrival_date = st.selectbox("Arrival Date", list(range(1, 32)))
market_segment_type = st.selectbox("Market Segment", ['Offline', 'Online', 'Corporate', 'Aviation', 'Complementary'])
repeated_guest = st.selectbox("Is a Repeated Guest?", [0, 1])
no_of_previous_cancellations = st.number_input("Previous Cancellations", min_value=0, max_value=20, value=0)
no_of_previous_bookings_not_canceled = st.number_input("Previous Non-Canceled Bookings", min_value=0, max_value=60, value=0)
avg_price_per_room = st.number_input("Average Price per Room", min_value=0.0, max_value=10000.0, value=100.0)
no_of_special_requests = st.slider("Number of Special Requests", 0, 5, 0)

# Run prediction when button is clicked
if st.button("Predict"):
    # Collect inputs into a DataFrame
    input_data = pd.DataFrame([{
        "no_of_adults": no_of_adults,
        "no_of_children": no_of_children,
        "no_of_weekend_nights": no_of_weekend_nights,
        "no_of_week_nights": no_of_week_nights,
        "type_of_meal_plan": type_of_meal_plan,
        "required_car_parking_space": required_car_parking_space,
        "room_type_reserved": room_type_reserved,
        "lead_time": lead_time,
        "arrival_year": arrival_year,
        "arrival_month": arrival_month,
        "arrival_date": arrival_date,
        "market_segment_type": market_segment_type,
        "repeated_guest": repeated_guest,
        "no_of_previous_cancellations": no_of_previous_cancellations,
        "no_of_previous_bookings_not_canceled": no_of_previous_bookings_not_canceled,
        "avg_price_per_room": avg_price_per_room,
        "no_of_special_requests": no_of_special_requests
    }])

    # Preprocess & predict
    X_processed = preprocessor.transform(input_data)
    pred = model.predict(X_processed)
    label = label_encoder.inverse_transform(pred)

    # Display result
    st.subheader("Prediction Result")
    st.success(f"📢 Booking Status: **{label[0]}**")
