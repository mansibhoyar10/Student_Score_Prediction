import os
import pickle
import numpy as np
import streamlit as st

# Set page configuration
st.set_page_config(page_title="Student Performance Predictor", layout="centered")

# App Header
st.title("🎓 Student Performance Predictor")
st.write("Enter the student details below to estimate the final score.")

MODEL_PATH = 'model(1).pkl'

# Load the model safely
@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, 'rb') as f:
            return pickle.load(f)
    else:
        st.error(f"Could not find '{MODEL_PATH}'. Please ensure your model pkl file is in the same directory.")
        return None

model = load_model()

if model is not None:
    # Create inputs for the 4 expected features
    st.header("📋 Input Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        hours_studied = st.number_input("Hours Studied", min_value=0.0, max_value=24.0, value=5.0, step=0.5)
        sleep_hours = st.number_input("Sleep Hours", min_value=0.0, max_value=24.0, value=7.0, step=0.5)
        
    with col2:
        attendance_percent = st.slider("Attendance Percent", min_value=0.0, max_value=100.0, value=85.0, step=1.0)
        previous_scores = st.number_input("Previous Scores", min_value=0.0, max_value=100.0, value=75.0, step=1.0)
        
    st.markdown("---")
    
    # Predict button
    if st.button("🔮 Predict Score", type="primary"):
        # Format the inputs exactly in the order the model expects
        features = np.array([[hours_studied, sleep_hours, attendance_percent, previous_scores]])
        
        # Run prediction
        try:
            prediction = model.predict(features)
            
            # Display results
            st.success(f"### Predicted Result: **{prediction[0]:.2f}**")
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
