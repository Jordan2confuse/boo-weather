import streamlit as st
import pandas as pd
import numpy as np
import json
from sklearn.ensemble import RandomForestClassifier

# Load model JSON
with open("random_forest_model.json", "r") as f:
    model_components = json.load(f)

# Retrieve important details (e.g., number of trees, mtry, etc.)
ntree = model_components["ntree"]
mtry = model_components["mtry"]
# Use the model formula or import a trained RandomForest model.

# Dummy function for predictions (replace with actual Random Forest usage)
def predict_weather(data):
    # Simulate a prediction (use your trained model here)
    return np.random.choice(["Rain", "Sunny", "Cloudy"], p=[0.3, 0.4, 0.3])

st.title("Weather Forecast Prediction")

# Input fields
temperature = st.number_input("Enter the Temperature:", min_value=-50, max_value=50, value=20)
humidity = st.number_input("Enter the Humidity (%):", min_value=0, max_value=100, value=50)
wind_speed = st.number_input("Enter the Wind Speed (km/h):", min_value=0, max_value=200, value=10)

# Prediction Button
if st.button("Predict Weather"):
    input_data = pd.DataFrame({
        "Temperature": [temperature],
        "Humidity": [humidity],
        "Wind Speed": [wind_speed]
    })
    
    # Call prediction function
    prediction = predict_weather(input_data)
    st.write(f"Predicted Weather: {prediction}")
