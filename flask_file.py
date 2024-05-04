from flask import Flask, request, jsonify
import joblib

# Initialize Flask application
app = Flask(__name__)

# Load trained model
model = joblib.load('best_model.pkl') 

# Define route for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Get input data from request
    data = request.json
    features = []
    for k, v in data.items():
        if k != 'id':
            features.append(v)
    # Make predictions using the loaded model
    predictions = model.predict([features])
    
    # Return predictions as JSON response
    return jsonify(predictions.tolist()[0])

# Run Flask application
if __name__ == '__main__':
    app.run(debug=True)
