
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.xception import preprocess_input
from PIL import Image, UnidentifiedImageError

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

model_path = 'D:/shayaan/script/cat_trained_model_Xception.h5'
model = tf.keras.models.load_model(model_path)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/classify', methods=['POST'])
def classify():
    try:
        if 'file' not in request.files:
            raise ValueError('No file part')

        file = request.files['file']

        if file.filename == '':
            raise ValueError('No selected file')

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            file.save(file_path)

            img = Image.open(file_path)
            img_width, img_height = 224, 224
            img = img.resize((img_width, img_height))

            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)

            predictions = model.predict(img_array)

            labels = ['Abyssinian', 'Bengal', 'Birman', 'Bombay', 'British_Shorthair']
            decoded_predictions = [(label, float(score)) for label, score in zip(labels, predictions[0])]

            top_prediction = max(decoded_predictions, key=lambda x: x[1])

            result_str = f"The detected breed of cat is: {top_prediction[0]} with accuracy {top_prediction[1]:.2f}"

            return jsonify({'result': result_str, 'class': top_prediction[0], 'confidence': float(top_prediction[1]), 'image_path': os.path.abspath(file_path)}), 200, {'Content-Type': 'application/json'}

        else:
            raise ValueError('File type not allowed')

    except Exception as e:
      return jsonify({'error': str(e), 'breed': 'Unknown Breed', 'confidence': 0})


if __name__ == '__main__':
    app.run(debug=True)