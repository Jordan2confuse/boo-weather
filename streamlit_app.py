import streamlit as st
import pandas as pd
import numpy as np
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter

# Load R model and preprocessing info
@st.cache_resource
def load_model():
    # Load the R model
    r = robjects.r
    model = r['readRDS']('rf_model.rds')
    preprocess_info = r['readRDS']('preprocessing_info.rds')
    feature_importance = pd.read_csv('feature_importance.csv')
    with open('feature_names.txt', 'r') as f:
        feature_names = [line.strip() for line in f]
    return model, preprocess_info, feature_importance, feature_names

# Make predictions using R model
def predict_r(model, data):
    with localconverter(robjects.default_converter + pandas2ri.converter):
        r_df = pandas2ri.py2rpy(data)
        predictions = robjects.r.predict(model, r_df)
        return pandas2ri.rpy2py(predictions)

# Initialize the app
st.title('Random Forest Classifier Prediction App')
st.write('Enter your features to get predictions')

try:
    model, preprocess_info, feature_importance, feature_names = load_model()

    # Create input fields for features
    input_data = {}
    for feature in feature_names:
        input_data[feature] = st.number_input(f'Enter {feature}:', value=0.0)

    # Create prediction button
    if st.button('Predict'):
        # Convert input to DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Make prediction
        prediction = predict_r(model, input_df)
        
        # Display prediction
        st.write('### Prediction')
        st.write(f'The predicted class is: {prediction[0]}')

    # Display feature importance plot
    st.write('### Feature Importance')
    importance_fig = st.bar_chart(
        feature_importance.set_index('Unnamed: 0')['MeanDecreaseGini']
    )

except Exception as e:
    st.error(f'Error loading model: {str(e)}')
