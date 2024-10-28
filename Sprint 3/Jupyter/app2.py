from flask import Flask, request, render_template
import joblib
import pandas as pd

# Initialize the Flask app
app = Flask(__name__)

# Load models and scaler
best_rf_model = joblib.load('best_random_forest_model.pkl')
best_svm_model = joblib.load('best_svm_model.pkl')
voting_clf = joblib.load('voting_classifier_model.pkl')
scaler = joblib.load('scaler.pkl')

# Home route to render the HTML form
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route for Random Forest
@app.route('/predict_rf', methods=['POST'])
def predict_rf():
    # Get data from form inputs
    input_data = {
        'Jitter': float(request.form['Jitter']),
        'Jitter(Abs)': float(request.form['JitterAbs']),
        'Jitter:RAP': float(request.form['JitterRAP']),
        'Jitter:PPQ5': float(request.form['JitterPPQ5']),
        'Jitter:DDP': float(request.form['JitterDDP']),
        'Shimmer': float(request.form['Shimmer']),
        'Shimmer(dB)': float(request.form['ShimmerDB']),
        'Shimmer:APQ3': float(request.form['ShimmerAPQ3']),
        'Shimmer:APQ5': float(request.form['ShimmerAPQ5']),
        'Shimmer:APQ11': float(request.form['ShimmerAPQ11']),
        'Shimmer:DDA': float(request.form['ShimmerDDA']),
        'NHR': float(request.form['NHR']),
        'HNR': float(request.form['HNR'])
    }

    # Create a DataFrame and scale the input data
    input_df = pd.DataFrame([input_data])
    input_scaled = scaler.transform(input_df)

    # Make a prediction with the Random Forest model
    prediction = best_rf_model.predict(input_scaled)

    # Return the prediction as a response
    result = "Parkinson's disease" if prediction[0] == 1 else "No Parkinson's disease"
    
    return render_template('result.html', prediction_text=f' {result}')

# Prediction route for SVM
@app.route('/predict_svm', methods=['POST'])
def predict_svm():
    # Get data from form inputs
    input_data = {
        'Jitter': float(request.form['Jitter']),
        'Jitter(Abs)': float(request.form['JitterAbs']),
        'Jitter:RAP': float(request.form['JitterRAP']),
        'Jitter:PPQ5': float(request.form['JitterPPQ5']),
        'Jitter:DDP': float(request.form['JitterDDP']),
        'Shimmer': float(request.form['Shimmer']),
        'Shimmer(dB)': float(request.form['ShimmerDB']),
        'Shimmer:APQ3': float(request.form['ShimmerAPQ3']),
        'Shimmer:APQ5': float(request.form['ShimmerAPQ5']),
        'Shimmer:APQ11': float(request.form['ShimmerAPQ11']),
        'Shimmer:DDA': float(request.form['ShimmerDDA']),
        'NHR': float(request.form['NHR']),
        'HNR': float(request.form['HNR'])
    }

    # Create a DataFrame and scale the input data
    input_df = pd.DataFrame([input_data])
    input_scaled = scaler.transform(input_df)

    # Make a prediction with the SVM model
    prediction = best_svm_model.predict(input_scaled)

    # Return the prediction as a response
    result = "Parkinson's disease" if prediction[0] == 1 else "No Parkinson's disease"
    
    return render_template('result.html', prediction_text=f'SVM Prediction: {result}')

# Prediction route for Voting Classifier
@app.route('/predict_voting', methods=['POST'])
def predict_voting():
    # Get data from form inputs
    input_data = {
        'Jitter': float(request.form['Jitter']),
        'Jitter(Abs)': float(request.form['JitterAbs']),
        'Jitter:RAP': float(request.form['JitterRAP']),
        'Jitter:PPQ5': float(request.form['JitterPPQ5']),
        'Jitter:DDP': float(request.form['JitterDDP']),
        'Shimmer': float(request.form['Shimmer']),
        'Shimmer(dB)': float(request.form['ShimmerDB']),
        'Shimmer:APQ3': float(request.form['ShimmerAPQ3']),
        'Shimmer:APQ5': float(request.form['ShimmerAPQ5']),
        'Shimmer:APQ11': float(request.form['ShimmerAPQ11']),
        'Shimmer:DDA': float(request.form['ShimmerDDA']),
        'NHR': float(request.form['NHR']),
        'HNR': float(request.form['HNR'])
    }

    # Create a DataFrame and scale the input data
    input_df = pd.DataFrame([input_data])
    input_scaled = scaler.transform(input_df)

    # Make a prediction with the Voting Classifier model
    prediction = voting_clf.predict(input_scaled)

    # Return the prediction as a response
    result = "Parkinson's disease" if prediction[0] == 1 else "No Parkinson's disease"
    
    return render_template('result.html', prediction_text=f'Voting Classifier Prediction: {result}')

if __name__ == "__main__":
    app.run(debug=True)



















'''# app.py
from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
from sklearn.preprocessing import RobustScaler

# Initialize the Flask app
app = Flask(__name__)

# Load models and scaler
best_rf_model = joblib.load('best_random_forest_model.pkl')
best_svm_model = joblib.load('best_svm_model.pkl')
voting_clf = joblib.load('voting_classifier_model.pkl')
scaler = joblib.load('scaler.pkl')

# Home route to render the HTML form
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # Get data from form inputs
    input_data = {
        'Jitter': float(request.form['Jitter']),
        'Jitter(Abs)': float(request.form['JitterAbs']),
        'Jitter:RAP': float(request.form['JitterRAP']),
        'Jitter:PPQ5': float(request.form['JitterPPQ5']),
        'Jitter:DDP': float(request.form['JitterDDP']),
        'Shimmer': float(request.form['Shimmer']),
        'Shimmer(dB)': float(request.form['ShimmerDB']),
        'Shimmer:APQ3': float(request.form['ShimmerAPQ3']),
        'Shimmer:APQ5': float(request.form['ShimmerAPQ5']),
        'Shimmer:APQ11': float(request.form['ShimmerAPQ11']),
        'Shimmer:DDA': float(request.form['ShimmerDDA']),
        'NHR': float(request.form['NHR']),
        'HNR': float(request.form['HNR'])
    }

    # Create a DataFrame and scale the input data
    input_df = pd.DataFrame([input_data])
    input_scaled = scaler.transform(input_df)

    # Make a prediction with the Voting Classifier
    prediction = voting_clf.predict(input_scaled)

    # Return the prediction as a response
    result = "Parkinson's disease" if prediction[0] == 1 else "No Parkinson's disease"
    
    return render_template('index.html', prediction_text=f'Prediction: {result}')

if __name__ == "__main__":
    app.run(debug=True)'''
