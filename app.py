import streamlit as st
from transformers import AutoTokenizer, TFAutoModelForTokenClassification

# Load model and tokenizer
model_name = "YourUsername/arabic-pos-model"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = TFAutoModelForTokenClassification.from_pretrained(model_name)

st.title("Arabic POS Tagger")
text = st.text_area("Enter Arabic text:")
if st.button("Tag POS") and text:
    tokens = text.split()
    inputs = tokenizer(tokens, is_split_into_words=True, return_tensors="tf", truncation=True)
    logits = model(**inputs).logits
    pred_ids = logits.argmax(axis=-1).numpy()[0]
    tags = [model.config.id2label[i] for i in pred_ids]
    st.write(list(zip(tokens, tags)))