
# sms_lstm_train_and_app.py

import os
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping

# Paramètres
MAX_WORDS = 10000
MAX_LEN = 100
MODEL_PATH = "spam_lstm_model.h5"

# Chargement et préparation des données
@st.cache_data
def load_and_prepare_data():
    df = pd.read_csv("https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv",
                     sep='\t', header=None, names=['label', 'message'])
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    return df

df = load_and_prepare_data()

# Tokenisation
tokenizer = Tokenizer(num_words=MAX_WORDS)
tokenizer.fit_on_texts(df['message'])
X = tokenizer.texts_to_sequences(df['message'])
X = pad_sequences(X, maxlen=MAX_LEN)
y = df['label'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Chargement ou entraînement du modèle
@st.cache_resource
def get_or_train_model():
    if os.path.exists(MODEL_PATH):
        model = load_model(MODEL_PATH)
    else:
        model = Sequential([
            Embedding(input_dim=MAX_WORDS, output_dim=128, input_length=MAX_LEN),
            LSTM(64, dropout=0.2, recurrent_dropout=0.2),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        early_stop = EarlyStopping(patience=2, restore_best_weights=True)
        model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, callbacks=[early_stop])
        model.save(MODEL_PATH)
    return model

model = get_or_train_model()

# Interface Streamlit
st.title("📩 Détection de SMS Spam avec LSTM")
st.markdown("Entrez un message texte pour vérifier s’il s’agit de spam ou non :")

user_input = st.text_area("Message texte", "Congratulations! You've won a free ticket to Bahamas. Reply now!")

def predict_sms(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=MAX_LEN)
    pred = model.predict(padded)[0][0]
    label = "Spam 📛" if pred > 0.5 else "Non-Spam ✅"
    return label, pred

if st.button("Analyser"):
    label, score = predict_sms(user_input)
    st.markdown(f"### Résultat : `{label}`")
    st.write(f"Score de confiance : **{score:.4f}**")
