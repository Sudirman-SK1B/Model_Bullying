import joblib
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

import tensorflow as tf
import matplotlib as mpl
from cycler import cycler
mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['lines.linestyle'] = '--'
import re
import string
import nltk
nltk.download('stopwords')
sns.despine()
plt.style.use("fivethirtyeight")
sns.set_style("darkgrid")
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

model = load_model('Model_Bullying.h5')

# Tokenisasi dan stemming
def tokenize_and_stem(text):
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    stemmer = PorterStemmer()
    stems = [stemmer.stem(t) for t in tokens]
    return stems

def normalize_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    stop_words = set(stopwords.words('indonesian'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# Fungsi untuk mendapatkan kategori berdasarkan range persentase
def get_category(percentage):
    if percentage < 0.33:
        return 'Ringan'
    elif 0.33 <= percentage < 0.67:
        return 'Sedang'
    else:
        return 'Berat'
    
def predict_bulliance(texts):
    # Normalisasi teks pada data uji
    normalized_texts = [normalize_text(text) for text in texts]

    # Representasi vektor menggunakan CountVectorizer dan TF-IDF Transformer
    vectorizer = joblib.load('vectorizer.pkl')
    tfidf_transformer = joblib.load('transformer.pkl')
    label_encoder = joblib.load('label_encoder.pkl')
    text_counts = vectorizer.transform(normalized_texts)
    text_tfidf = tfidf_transformer.transform(text_counts)

    threshold = 50.00

    # Prediksi label pada data uji baru
    predictions = model.predict(text_tfidf.toarray())

    # Menampilkan hasil prediksi
    confidence = np.max(predictions) * 100
    category = get_category(confidence / 100)  # Konversi persentase ke range 0-1
    predicted_label = "Bullying" if confidence > threshold else "Non-Bullying"

    print(f'Teks: {texts[0]}')
    print(f'Prediksi: {predicted_label}')
    print(f'Persentase: {confidence:.2f}%')
    print(f'Kategori: {category}')
    print()
    return predicted_label