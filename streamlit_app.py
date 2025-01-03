import streamlit as st
import pandas as pd

st.title(' 🌀🌡️☁️ Weather Alert System')

st.info('Get your latest weather updates!')

df = pd.read_csv('https://raw.githubusercontent.com/Jordan2confuse/boo-weather/refs/heads/master/weather_prediction_dataset.csv')
df
