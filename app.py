import streamlit as st
import pickle
import numpy as np
import sklearn

# Load the preprocessor and model using pickle
try:
    with open('Preprocessor.pkl', 'rb') as f:
        preprocessor = pickle.load(f)
    with open('Model.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error("Preprocessor or model file not found.")
    st.stop()

st.title("Agricultural Yield Prediction")
st.info("Note: The prediction may not be entirely accurate due to the complexity and variability of agricultural factors. Trained with data from 1990 to 2013 only.")
st.header("Input Parameters")

# List of allowed crop names
allowed_crops = [
    'Maize', 'Potatoes', 'Rice, paddy', 'Sorghum', 'Soybeans', 'Wheat',
    'Cassava', 'Sweet potatoes', 'Plantains and others', 'Yams'
]

# Display the allowed crop names
st.subheader("Allowed Crop Names")
st.write(", ".join(allowed_crops))

# Input fields
Year = st.number_input('Year', min_value=1900, max_value=2100, step=1, help="Enter the year of prediction")
average_rain_fall_mm_per_year = st.number_input('Average Rainfall (mm/year)', help="Enter the average annual rainfall in mm")
pesticides_tonnes = st.number_input('Pesticides (tonnes)', help="Enter the amount of pesticides used in tonnes")
avg_temp = st.number_input('Average Temperature (°C)', help="Enter the average annual temperature in °C")
Area = st.text_input('Country', help="Enter the name of the country")
Item = st.selectbox('Crop Name', allowed_crops, help="Select the crop name from the allowed list")

# Prediction button
if st.button('Predict'):
    try:
        # Capitalize the first letter of Area
        Area = Area.capitalize()

        # Prepare the feature array
        feature = np.array([[Year, average_rain_fall_mm_per_year, pesticides_tonnes, avg_temp, Area, Item]])
        
        # Process the data using the preprocessor
        processed_data = preprocessor.transform(feature)
        
        # Make the prediction
        prediction = model.predict(processed_data)
        
        # Display the prediction result
        st.success(f"Predicted Yield: {prediction[0] / 10.0} Kilograms per Hectare")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
