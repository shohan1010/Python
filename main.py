import subprocess
import sys

# Install flask_cors using pip if not already installed
try:
    import flask_cors
except ImportError:
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'flask_cors'])

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

app = Flask(__name__)
CORS(app)

MODEL = tf.keras.models.load_model("test1_demo.h5")
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

@app.route('/ping', methods=['GET'])
def ping():
    return "Hello, I am alive"

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"})

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected file"})

    try:
        image = read_file_as_image(file.read())
        img_batch = np.expand_dims(image, 0)

        predictions = MODEL.predict(img_batch)

        predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
        confidence = float(np.max(predictions[0]))

        return jsonify({
            'class': predicted_class,
            'confidence': confidence
        })

    except Exception as e:
        return jsonify({"error": str(e)})


def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image


if __name__ == '__main__':
    app.run(debug=True)
