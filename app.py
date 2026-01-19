from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
import pickle

app = Flask(__name__)

model = tf.keras.models.load_model('wine_model.keras', compile=False)
with open('wine_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Human-readable names for the cultivators
CULTIVATORS = ["Barolo", "Grignolino", "Barbera"]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        # The dataset has 13 features; for simplicity, we'll take the first 4 key ones
        # and fill the rest with average values for the demo
        user_input = [float(data[f]) for f in ['alcohol', 'malic_acid', 'ash', 'alcalinity']]
        full_input = user_input + [13.0] * 9 # Padding remaining 9 features
        
        raw_input = np.array([full_input])
        scaled_input = scaler.transform(raw_input)
        
        # Get probabilities for all 3 classes
        predictions = model.predict(scaled_input)
        class_index = np.argmax(predictions) # Pick the highest probability index
        
        return jsonify({
            'success': True,
            'origin': CULTIVATORS[class_index],
            'confidence': f"{np.max(predictions) * 100:.1f}%"
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)