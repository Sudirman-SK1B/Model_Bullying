# Gunakan base image Python
FROM python:3.8-slim

# Set working directory di dalam container
WORKDIR /app

# Tambahkan file aplikasi dan dependencies
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Tambahkan file aplikasi ke dalam container
COPY app.py .

# Expose port yang digunakan oleh aplikasi
EXPOSE 8000

# Command untuk menjalankan aplikasi Flask
CMD ["python", "app.py"]
