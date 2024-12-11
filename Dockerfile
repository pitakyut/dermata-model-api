# Gunakan image Python slim
FROM python:3.9-slim

# Tentukan direktori kerja
WORKDIR /app

# Salin file requirements.txt
COPY requirements.txt .

# Instal pustaka sistem yang diperlukan
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Instal semua dependensi Python
RUN pip install --no-cache-dir -r requirements.txt

# Pastikan gunicorn diinstal secara eksplisit
RUN pip install gunicorn

# Salin semua file aplikasi
COPY . .

# Menentukan perintah untuk menjalankan aplikasi
CMD ["gunicorn", "-b", "0.0.0.0:8080", "app:app"]

# Membuka port 8080
EXPOSE 8080
