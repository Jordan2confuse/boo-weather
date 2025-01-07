import streamlit as st
import pandas as pd
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter

def load_model():
    # Load the R model
    r = robjects.r
    model = r.readRDS("random_forest_model.rds")
    return model

def predict(model, input_data):
    # Convert Python DataFrame to R DataFrame
    with localconverter(robjects.default_converter + pandas2ri.converter):
        r_input = pandas2ri.py2rpy(input_data)
    
    # Make predictions
    r_pred = robjects.r.predict(model, r_input)
    
    # Convert R predictions back to Python
    with localconverter(robjects.default_converter + pandas2ri.converter):
        predictions = pandas2ri.rpy2py(r_pred)
    
    return predictions

def main():
    st.title("Random Forest Classifier Prediction")
    
    # Create input fields for your features
    st.header("Enter Features:")
    
    # Add input fields for each feature (modify according to your model)
    feature1 = st.number_input("Feature 1", value=0.0)
    feature2 = st.number_input("Feature 2", value=0.0)
    # Add more features as needed
    
    # Create a button for prediction
    if st.button("Predict"):
        # Create DataFrame from inputs
        input_data = pd.DataFrame({
            'feature1': [feature1],
            'feature2': [feature2]
            # Add more features as needed
        })
        
        # Load model and make prediction
        model = load_model()
        prediction = predict(model, input_data)
        
        # Display prediction
        st.success(f"Prediction: {prediction[0]}")

if __name__ == "__main__":
    main()
