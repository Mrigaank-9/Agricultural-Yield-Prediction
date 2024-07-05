import streamlit as st
import joblib
import numpy as np

try:
    preprocessor = joblib.load('Preprocessor.pkl')
    model = joblib.load('Model.pkl')
except FileNotFoundError:
    st.error("Preprocessor or model file not found.")
    st.stop()

st.title("Agricultural Yield Prediction")
st.info("Note: The prediction may not be entirely accurate due to the complexity and variability of agricultural factors. Trained with data from 1990 to 2013 only")
st.header("Input Parameters")

Year = st.number_input('Year', min_value=1900, max_value=2100, step=1)
average_rain_fall_mm_per_year = st.number_input('Average Rainfall (mm/year)')
pesticides_tonnes = st.number_input('Pesticides (tonnes)')
avg_temp = st.number_input('Average Temperature (°C)')
Area = st.text_input('Country')
Item = st.text_input('Crop Name')

# Prediction button
if st.button('Predict'):
    try:
        feature = np.array([[Year, average_rain_fall_mm_per_year, pesticides_tonnes, avg_temp, Area, Item]])
        processed_data = preprocessor.transform(feature)
        prediction = model.predict(processed_data)

        st.success(f"Predicted Yield: {prediction[0]} HectoGram per Hectare")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
