from flask import Flask, jsonify, request
import joblib
from server import *
app = Flask(__name__)

#Endpoint untuk halaman home
@app.route('/')
def home():
    return 'Welcome To Bullience Apps'
# Endpoint untuk melakukan prediksi dengan model pertama
@app.route('/predict/bullying', methods=['POST'])
def predict_model1():
    try:
        data = request.get_json()
        # Lakukan prediksi menggunakan model pertama
        prediction = predict_bulliance([data['text']]) # Ubah sesuai kebutuhan

        return jsonify({"prediction": str(prediction)})
    except Exception as e:
        return jsonify({"error": str(e)})

# Endpoint untuk melakukan prediksi dengan model kedua
@app.route('/predict/model2', methods=['POST'])
def predict_model2():
    try:
        data = request.get_json()
        # Lakukan prediksi menggunakan model kedua
        prediction = model2.predict([data['text']])  # Ubah sesuai kebutuhan

        return jsonify({"prediction": str(prediction)})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
