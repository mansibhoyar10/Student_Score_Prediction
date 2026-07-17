import os
import pickle
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

# Define the expected model file path
MODEL_PATH = 'model(1).pkl'

# Load the model at startup
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
else:
    raise FileNotFoundError(f"Could not find '{MODEL_PATH}'. Please ensure your model pkl file is in the same directory.")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse the incoming JSON request
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No input data provided. Expected a JSON object.'}), 400

        # Define the expected features based on the model metadata
        required_features = ['hours_studied', 'sleep_hours', 'attendance_percent', 'previous_scores']
        
        # Validate that all features are present
        missing_features = [feat for feat in required_features if feat not in data]
        if missing_features:
            return jsonify({'error': f'Missing features in request: {missing_features}'}), 400

        # Extract values in the exact order the model expects
        features = [
            float(data['hours_studied']),
            float(data['sleep_hours']),
            float(data['attendance_percent']),
            float(data['previous_scores'])
        ]

        # Convert to a 2D array shape for the scikit-learn model
        input_array = np.array([features])

        # Make prediction
        prediction = model.predict(input_array)

        # Return the result as JSON (extracting the first scalar from the array output)
        return jsonify({
            'prediction': float(prediction[0])
        })

    except ValueError as val_err:
        return jsonify({'error': f'Invalid value type: {str(val_err)}. Values must be numbers.'}), 400
    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

if __name__ == '__main__':
    # Run the Flask app on localhost port 5000
    app.run(host='0.0.0.0', port=5000, debug=True)
