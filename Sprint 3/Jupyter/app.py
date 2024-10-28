import pandas as pd
import joblib
import streamlit as st
from sklearn.preprocessing import StandardScaler

# Load the models and scaler
rf_model = joblib.load('random_forest_model.pkl')  # Replace with your saved model path
svm_model = joblib.load('svm_model.pkl')  # Replace with your saved model path
scaler = joblib.load('scaler.pkl')  # Replace with your saved scaler path

# Streamlit app title
st.title("Parkinson's Disease Detection")

# Input fields
st.header("Input Features")
jitter = st.number_input("Jitter", value=0.620, format="%.6f")
jitter_abs = st.number_input("Jitter(Abs)", value=0.00003000, format="%.8f")
jitter_rap = st.number_input("Jitter:RAP", value=0.250, format="%.3f")
jitter_ppq5 = st.number_input("Jitter:PPQ5", value=0.22, format="%.2f")
jitter_ddp = st.number_input("Jitter:DDP", value=0.750, format="%.3f")
shimmer = st.number_input("Shimmer", value=3.600, format="%.3f")
shimmer_db = st.number_input("Shimmer(dB)", value=0.485, format="%.3f")
shimmer_apq3 = st.number_input("Shimmer:APQ3", value=1.70, format="%.2f")
shimmer_apq5 = st.number_input("Shimmer:APQ5", value=2.10, format="%.2f")
shimmer_apq11 = st.number_input("Shimmer:APQ11", value=3.100, format="%.3f")
shimmer_dda = st.number_input("Shimmer:DDA", value=5.200, format="%.3f")
nhr = st.number_input("NHR", value=0.052000, format="%.6f")
hnr = st.number_input("HNR", value=17.400, format="%.3f")

# Button for prediction
if st.button("Predict"):
    # Prepare input for prediction
    input_data = {
        'Jitter': jitter,
        'Jitter(Abs)': jitter_abs,
        'Jitter:RAP': jitter_rap,
        'Jitter:PPQ5': jitter_ppq5,
        'Jitter:DDP': jitter_ddp,
        'Shimmer': shimmer,
        'Shimmer(dB)': shimmer_db,
        'Shimmer:APQ3': shimmer_apq3,
        'Shimmer:APQ5': shimmer_apq5,
        'Shimmer:APQ11': shimmer_apq11,
        'Shimmer:DDA': shimmer_dda,
        'NHR': nhr,
        'HNR': hnr
    }

    input_df = pd.DataFrame([input_data])
    input_scaled = scaler.transform(input_df)

    # Make predictions with both models
    prediction_rf = rf_model.predict(input_scaled)
    prediction_svm = svm_model.predict(input_scaled)

    # Display results
    st.subheader("Predictions")
    st.write("Random Forest Prediction:", "Parkinson's disease" if prediction_rf[0] == 1 else "No Parkinson's disease")
    st.write("SVM Prediction:", "Parkinson's disease" if prediction_svm[0] == 1 else "No Parkinson's disease")

# Run the Streamlit app
if __name__ == "__main__":
    #st.write("Run this app by executing 'streamlit run app.py' in your terminal.")
        st.write(" ")

