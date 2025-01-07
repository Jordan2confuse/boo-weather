import streamlit as st
import pandas as pd
import os
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter

# Enable R/Python conversion
pandas2ri.activate()

# Add error handling for R_HOME
if not os.environ.get('R_HOME'):
    st.error("R_HOME environment variable is not set. Please set it to your R installation path.")
    st.stop()

try:
    # Load the R readRDS function
    read_rds = robjects.r['readRDS']
    
    # Load the model with error handling
    model_path = r"C:\Users\Jordan Boo\OneDrive\Desktop\random_forest_model.rds.R"
    if not os.path.exists(model_path):
        st.error(f"Model file not found at: {model_path}")
        st.stop()
        
    model = read_rds(model_path)
    
    # Title of the app
    st.title("Weather Alert System")

    # Input fields - Update these names to match your actual model features
    feature1 = st.number_input("Temperature (°C)")
    feature2 = st.number_input("Humidity (%)")
    feature3 = st.number_input("Wind Speed (km/h)")
    feature4 = st.number_input("Pressure (hPa)")

    # Create a DataFrame from user inputs
    input_data = pd.DataFrame({
        "temperature": [feature1],
        "humidity": [feature2],
        "wind_speed": [feature3],
        "pressure": [feature4]
    })

    # Display user inputs
    st.subheader("User Input Data:")
    st.write(input_data)

    # Predict button
    if st.button("Predict"):
        try:
            # Convert Python DataFrame to R DataFrame
            with localconverter(robjects.default_converter + pandas2ri.converter):
                r_input_data = robjects.conversion.py2rpy(input_data)
            
            # Make prediction
            r_predict = robjects.r['predict']
            prediction = r_predict(model, r_input_data)
            
            # Convert R prediction back to Python
            with localconverter(robjects.default_converter + pandas2ri.converter):
                py_prediction = robjects.conversion.rpy2py(prediction)
            
            st.subheader("Prediction:")
            st.write(py_prediction[0])
            
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")

except Exception as e:
    st.error(f"Error loading model: {str(e)}")


