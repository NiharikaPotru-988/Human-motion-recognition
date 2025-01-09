import os
import pandas as pd
from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# Load the trained model
model = load_model('human_action_recognition_model.keras')

# Load the training dataset
train_csv_path = "Training_set.csv"
train_df = pd.read_csv(train_csv_path)

# Create a label map from the training dataset
label_map = {index: label for index, label in enumerate(train_df['label'].astype('category').cat.categories)}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']
    img = Image.open(io.BytesIO(file.read()))
    img = img.resize((128, 128))  # Resize to the input size expected by your model
    img_array = np.array(img) / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=-1)[0]

    # Get the predicted label from the label map
    predicted_label = label_map.get(predicted_class, 'Unknown')

    return jsonify({'label': predicted_label})

if __name__ == '__main__':
    app.run(debug=True)