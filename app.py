import streamlit as st
from transformers import pipeline, AutoTokenizer, TFAutoModelForTokenClassification
import tensorflow as tf
import os

# Configure Kaggle-compatible settings
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_KERAS"] = "1"
pip install -q tf-keras==2.15.0 transformers==4.41.0 sentencepiece streamlit pyconll


# Configure Streamlit page
st.set_page_config(
    page_title="Arabic POS Tagger",
    page_icon="📚",
    layout="wide"
)

# Custom CSS for RTL support
st.markdown("""
<style>
.rtl-text {
    direction: rtl;
    text-align: right;
    font-size: 18px;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner="Loading POS model...")
def load_pos_model():
    # Explicitly load TensorFlow model
    model = TFAutoModelForTokenClassification.from_pretrained("MariamOsama3/Mariam_classifer2")
    tokenizer = AutoTokenizer.from_pretrained("MariamOsama3/Mariam_classifer2")
    
    return pipeline(
        "ner",
        model=model,
        tokenizer=tokenizer,
        framework="tf",  # Explicit TensorFlow framework
        aggregation_strategy="simple"
    )

# App interface
st.title("📖 Arabic Part-of-Speech Tagger")
input_text = st.text_area("Enter Arabic Text:", height=150, value="برلين ترفض حصول شركة امريكية على رخصة")

if st.button("Analyze Text"):
    if not input_text.strip():
        st.warning("Please enter Arabic text")
    else:
        with st.spinner("Analyzing..."):
            try:
                classifier = load_pos_model()
                results = classifier(input_text)
                
                # Display results
                st.subheader("Results")
                st.markdown('<div class="rtl-text">', unsafe_allow_html=True)
                for entity in results:
                    st.write(f"{entity['word']} - {entity['entity']} (confidence: {entity['score']:.2f})")
                st.markdown('</div>', unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.stop()
