import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv('Dataset_B_Hotel.csv')
    return df

df = load_data()
st.title("Hotel Booking Cancellation Prediction (XGBoost)")

# Drop Booking_ID column
df.drop(columns=['Booking_ID'], inplace=True)

# Handle missing values
df['type_of_meal_plan'].fillna('Not Selected', inplace=True)
df['required_car_parking_space'].fillna(0, inplace=True)

# Encode target variable
le_status = LabelEncoder()
df['booking_status'] = le_status.fit_transform(df['booking_status'])  # 1 = Not Canceled, 0 = Canceled

# Separate features and target
X = df.drop(columns=['booking_status'])
y = df['booking_status']

# Encode categorical features
cat_cols = X.select_dtypes(include='object').columns
le_dict = {}
for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    le_dict[col] = le  # Store encoder for user input

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost model
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
model.fit(X_train, y_train)

# Accuracy
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
st.write(f"Model Accuracy: {acc:.2%}")

# User input form
st.header("Enter Booking Details")

def user_input():
    adults = st.number_input('Number of Adults', min_value=0, max_value=10, value=2)
    children = st.number_input('Number of Children', min_value=0, max_value=10, value=0)
    weekend_nights = st.number_input('Weekend Nights', min_value=0, max_value=10, value=1)
    week_nights = st.number_input('Week Nights', min_value=0, max_value=20, value=2)
    meal_plan = st.selectbox('Meal Plan', ['Meal Plan 1', 'Not Selected', 'Meal Plan 2', 'Meal Plan 3'])
    parking = st.selectbox('Car Parking Space', [0, 1])
    room_type = st.selectbox('Reserved Room Type', ['Room_Type 1', 'Room_Type 2', 'Room_Type 3', 'Room_Type 4',
                                                    'Room_Type 5', 'Room_Type 6', 'Room_Type 7'])
    lead_time = st.slider('Lead Time (days)', 0, 500, 30)
    year = st.selectbox('Arrival Year', [2017, 2018])
    month = st.selectbox('Arrival Month', list(range(1, 13)))
    date = st.selectbox('Arrival Date', list(range(1, 32)))
    market_segment = st.selectbox('Market Segment Type', ['Offline', 'Online', 'Corporate', 'Aviation', 'Complementary'])
    repeat_guest = st.selectbox('Is Repeat Guest?', [0, 1])
    prev_canceled = st.number_input('Previous Cancellations', 0, 20, 0)
    prev_not_canceled = st.number_input('Previous Bookings Not Canceled', 0, 60, 0)
    avg_price = st.number_input('Average Room Price', 0.0, 500.0, 100.0)
    special_req = st.number_input('Number of Special Requests', 0, 5, 0)

    input_dict = {
        'no_of_adults': adults,
        'no_of_children': children,
        'no_of_weekend_nights': weekend_nights,
        'no_of_week_nights': week_nights,
        'type_of_meal_plan': meal_plan,
        'required_car_parking_space': parking,
        'room_type_reserved': room_type,
        'lead_time': lead_time,
        'arrival_year': year,
        'arrival_month': month,
        'arrival_date': date,
        'market_segment_type': market_segment,
        'repeated_guest': repeat_guest,
        'no_of_previous_cancellations': prev_canceled,
        'no_of_previous_bookings_not_canceled': prev_not_canceled,
        'avg_price_per_room': avg_price,
        'no_of_special_requests': special_req
    }

    return pd.DataFrame([input_dict])

user_df = user_input()

# Encode user input
for col in cat_cols:
    if col in user_df.columns:
        user_df[col] = le_dict[col].transform(user_df[col])

user_df = user_df[X_train.columns]
user_df = user_df.astype(X_train.dtypes.to_dict())

# Prediction
if st.button('Predict'):
    prediction = model.predict(user_df)[0]
    label = le_status.inverse_transform([prediction])[0]
    st.success(f"Prediction: **{label}**")
