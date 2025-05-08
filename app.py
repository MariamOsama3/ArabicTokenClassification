import streamlit as st
from utils import load_model, predict

st.title("Arabic Text Classification 🏷️")
st.markdown("Enter an Arabic sentence to classify it into predefined categories.")

text_input = st.text_area("Enter Arabic text here:")

if text_input:
    model, vectorizer = load_model()
    prediction = predict(text_input, model, vectorizer)
    st.success(f"Predicted Category: **{prediction}**")
