import streamlit as st
import joblib
import numpy as np

# Load the saved model and scaler from the subfolder
model = joblib.load('model/wine_cultivar_model.pkl')
scaler = joblib.load('model/scaler.pkl')

st.title("🍷 Wine Cultivar Prediction")

# User inputs
col1, col2 = st.columns(2)
with col1:
    alc = st.number_input("Alcohol", 11.0, 15.0, 13.0)
    mag = st.number_input("Magnesium", 70.0, 160.0, 100.0)
    flav = st.number_input("Flavanoids", 0.0, 5.0, 2.0)
with col2:
    col_int = st.number_input("Color Intensity", 1.0, 13.0, 5.0)
    hue = st.number_input("Hue", 0.0, 2.0, 1.0)
    pro = st.number_input("Proline", 200.0, 1700.0, 700.0)

if st.button("Predict Origin"):
    inputs = np.array([[alc, mag, flav, col_int, hue, pro]])
    scaled_inputs = scaler.transform(inputs)
    prediction = model.predict(scaled_inputs)
    
    st.success(f"Result: Cultivar {prediction[0] + 1}")