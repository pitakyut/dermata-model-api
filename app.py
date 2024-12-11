import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import numpy as np
from flask import Flask, request, jsonify
import tensorflow as tf
import tensorflow as tf
from firebase_admin import credentials, initialize_app
from PIL import Image
from google.cloud import storage
from google.cloud import firestore

# Inisialisasi aplikasi Flask
app = Flask(__name__)

MODEL = tf.keras.models.load_model('./dermata_inception_v3.h5', compile=False)
BUCKET_NAME =  'dermatapy-444204.appspot.com'

cred = credentials.Certificate("./dermata-444204-2c80dc7e1278.json")
initialize_app(cred, {"projectId": "dermata-444204"})
db = firestore.Client(database="dermata")

# Daftar label
labels = [
    "Acne", "Blackheads", "Dark Spots", "Dry Skin",
    "Eye Bags", "Normal Skin", "Oily Skin", "Pores",
    "Redness", "Wrinkles"
]

@app.route('/')
def index():
    return jsonify({"message": "Skin Diagnosis API is running!"})

def preprocess_image(image_file):
    try:
        img = Image.open(image_file).convert('RGB') 
        img = img.resize((540, 540))  # Resize image to 540x540 instead of 224x224
        img_array = np.array(img) / 255.0  # Normalize image
        img_array = np.expand_dims(img_array, axis=0).astype(np.float32)  # Add batch dimension
        return img_array
    except Exception as e:
        raise ValueError(f"Error processing image: {str(e)}")
    
@app.route('/predict', methods=['POST'])
def predict_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        file.seek(0)
        input_data = preprocess_image(file)

        predictions = MODEL.predict(input_data)
        predicted_label = np.argmax(predictions)
        confidence = predictions[0][predicted_label]
        label_name = labels[predicted_label]

        doc_ref = db.collection('predictions').document()
        doc_ref.set({
            "label_name": label_name,
            "confidence": float(confidence),
        })

        return jsonify({
            "prediction": label_name,
            "confidence": float(confidence),
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500