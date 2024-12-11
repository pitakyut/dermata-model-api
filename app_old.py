import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import numpy as np
import cv2
from flask import Flask, request, jsonify, session
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from firebase_admin import credentials, initialize_app
import requests
from google.cloud import storage
from google.cloud import firestore
import tempfile
import logging

# Inisialisasi aplikasi Flask
app = Flask(__name__)

# Nama bucket dan model file di Google Cloud Storage
BUCKET_NAME =  'dermatapy-444204.appspot.com'
MODEL_PATH = 'Dermata_inceptionV3_V3.h5'  

cred = credentials.Certificate("./dermata-444204-2c80dc7e1278.json")
initialize_app(cred, {"projectId": "dermata-444204"})
db = firestore.Client(database="dermata")

# Fungsi untuk mendownload model dari Google Cloud Storage
def download_model_from_gcs(bucket_name, model_path, local_path='model.h5'):
    try:
        # Menginisialisasi klien Google Cloud Storage
        storage_client = storage.Client()

        # Mengakses bucket dan file model
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(model_path)

        # Mendownload model ke path lokal
        blob.download_to_filename(local_path)

        print(f"Model downloaded from GCS: {model_path}")
    except Exception as e:
        raise ValueError(f"Error downloading model from GCS: {str(e)}")

# Memuat model yang sudah diunduh dari GCS
def load_model_from_gcs():
    # Jika model belum ada secara lokal, download model terlebih dahulu
    if not os.path.exists(MODEL_PATH):
        download_model_from_gcs(BUCKET_NAME, MODEL_PATH)

    # Memuat model
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

# Daftar label
labels = [
    "Acne", "Blackheads", "Dark Spots", "Dry Skin",
    "Eye Bags", "Normal Skin", "Oily Skin", "Pores",
    "Redness", "Wrinkles"
]

# Fungsi untuk memuat dan memproses gambar
def load_and_process_image(img_path, target_size=(540, 540, 3)):
    try:
        img = image.load_img(img_path, target_size=target_size[:2])
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = tf.keras.applications.inception_v3.preprocess_input(img_array)
        return img_array
    except Exception as e:
        raise ValueError(f"Error in processing image {img_path}: {str(e)}")

from google.cloud import firestore

# Fungsi untuk menyimpan hasil prediksi ke Firestore
def save_prediction_to_firestore(prediction_data):
    try:
        # Mengakses koleksi Firestore bernama 'predictions'
        predictions_ref = db.collection('predictions')
        
        # Menambahkan dokumen baru dengan data prediksi
        predictions_ref.add(prediction_data)
        print("Prediction saved to Firestore.")
    except Exception as e:
        print(f"Error saving prediction to Firestore: {str(e)}")

@app.route('/')
def index():
    return jsonify({"message": "Skin Diagnosis API is running!"})
    
@app.route('/predict', methods=['POST'])
def predict_file():
    if not os.path.exists('temp_image.jpg'):
        return jsonify({"error": "No uploaded file for prediction"}), 400

    try:
        # Memproses gambar
        img_array = load_and_process_image('temp_image.jpg', target_size=(540, 540, 3))

        # Memuat model
        model = load_model_from_gcs()

        # Prediksi menggunakan model
        predictions = model.predict(img_array)

        # Menentukan label yang diprediksi
        predicted_labels = (predictions >= 0.5).astype(int)

        # Menggabungkan hasil prediksi
        result = []
        prediction_details = []  # Menyimpan detail untuk Firestore

        for i in range(len(labels)):
            if predicted_labels[0][i] == 1:
                label = f"{labels[i]} ({predictions[0][i] * 100:.2f}%)"
                result.append(label)
                prediction_details.append({
                    "label": labels[i],
                    "confidence": predictions[0][i] * 100,
                })

        # Mengembalikan hasil
        if result:
            # Simpan hasil prediksi ke Firestore
            prediction_data = {
                "predictions": prediction_details,
                "result": result,
                "timestamp": firestore.SERVER_TIMESTAMP  # Menyimpan timestamp saat data disimpan
            }
            save_prediction_to_firestore(prediction_data)

            return jsonify({"diagnosis": result})
        else:
            return jsonify({"diagnosis": "No skin issues detected"})

    except Exception as e:
        return jsonify({"error": f"An error occurred during prediction: {str(e)}"}), 500


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