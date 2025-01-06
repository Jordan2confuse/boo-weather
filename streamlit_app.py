import streamlit as st
import subprocess

# Run R script using subprocess
subprocess.run(["Rscript", "Weather_FYP.R"])

st.title(' 🌀🌡️☁️ Weather Alert System')

st.info('Get your latest weather updates!')

library(streamlit)
library(randomForest)

# Load the saved model
rf_model <- readRDS("random_forest_model.rds")

# Streamlit UI
st$title("Weather Alert System")

# Input fields (replace with actual feature names)
feature1 <- st$number_input("Feature 1")
feature2 <- st$number_input("Feature 2")
feature3 <- st$number_input("Feature 3")
feature4 <- st$number_input("Feature 4")

# Predict button
if (st$button("Predict")) {
  input_data <- data.frame(Feature1 = feature1,
                           Feature2 = feature2,
                           Feature3 = feature3,
                           Feature4 = feature4)
  
  prediction <- predict(rf_model, input_data)
  
  st$subheader("Prediction:")
  st$write(as.character(prediction))
}




