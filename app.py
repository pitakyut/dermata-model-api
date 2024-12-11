import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import numpy as np
import cv2
from flask import Flask, request, jsonify, session
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tensorflow as tf
from firebase_admin import credentials, initialize_app
from PIL import Image
import requests
from google.cloud import storage
from google.cloud import firestore
import tempfile
import logging

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


# Route untuk mengunggah gambar
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    # Simpan gambar sementara
    img_path = 'temp_image.jpg'
    file.save(img_path)

    return jsonify({"message": "File uploaded successfully"}), 200

# Route untuk mengambil gambar ulang dari kamera
@app.route('/retry_camera', methods=['POST'])
def retry_camera():
    try:
        # Inisialisasi kamera
        camera = cv2.VideoCapture(0)  # Membuka kamera default
        ret, frame = camera.read()

        if not ret:
            return jsonify({"error": "Failed to capture image from camera"}), 500

        # Simpan gambar sementara untuk pengambilan ulang
        img_path = 'temp_camera_image.jpg'
        cv2.imwrite(img_path, frame)

        camera.release()  # Pastikan kamera dilepas setelah selesai

        return jsonify({"message": "Photo retaken successfully"}), 200

    except Exception as e:
        return jsonify({"error": f"Error retrying camera: {str(e)}"}), 500
    finally:
        if 'camera' in locals():
            camera.release()

# Route untuk menghapus gambar sementara
@app.route('/delete_photo', methods=['DELETE'])
def delete_photo():
    try:
        # File gambar sementara yang digunakan untuk prediksi
        img_path_file = 'temp_image.jpg'
        img_path_camera = 'temp_camera_image.jpg'

        # Menghapus file gambar jika ada
        if os.path.exists(img_path_file):
            os.remove(img_path_file)
        if os.path.exists(img_path_camera):
            os.remove(img_path_camera)

        return jsonify({"message": "Photos deleted successfully"}), 200
    except Exception as e:
        return jsonify({"error": f"Error deleting photos: {str(e)}"}), 500
    
# Menjalankan server Flask
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)