# utils.py
import joblib
import re
import string

def clean_text(text):
    # Basic Arabic text cleaning
    text = re.sub(r'[{}]+'.format(string.punctuation), '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def load_model(model_path="model/classifier.pkl", vectorizer_path="model/vectorizer.pkl"):
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer

def predict(text, model, vectorizer):
    cleaned = clean_text(text)
    vect_text = vectorizer.transform([cleaned])
    prediction = model.predict(vect_text)
    return prediction[0]
