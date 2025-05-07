import os
import streamlit as st
from transformers import pipeline, AutoTokenizer, TFAutoModelForTokenClassification

# --- Critical Configuration Section ---
os.environ["TF_USE_LEGACY_KERAS"] = "1"  # Essential for Keras 3 compatibility
os.environ["TF_KERAS"] = "1"             # Force TF Keras implementation
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Force CPU-only mode for deployment

# --- Streamlit Configuration ---
st.set_page_config(
    page_title="Arabic POS Tagger",
    page_icon="📚",
    layout="wide"
)

# Enhanced RTL styling with proper HTML rendering
st.markdown("""
<style>
.rtl-container {
    direction: rtl;
    text-align: right;
    font-family: 'Arial', sans-serif;
}
.entity-box {
    padding: 8px;
    margin: 4px;
    border-radius: 4px;
    background-color: #f0f2f6;
    display: inline-block;
}
</style>
""", unsafe_allow_html=True)

# --- Model Loading with Enhanced Error Handling ---
@st.cache_resource(show_spinner="Loading POS model...")
def load_pos_model():
    try:
        # Explicitly disable PyTorch
        os.environ["DISABLE_TORCH"] = "1"
        
        model = TFAutoModelForTokenClassification.from_pretrained(
            "MariamOsama3/Mariam_classifer2",
            from_pt=False  # Ensure TensorFlow model loading
        )
        tokenizer = AutoTokenizer.from_pretrained("MariamOsama3/Mariam_classifer2")
        
        return pipeline(
            "ner",
            model=model,
            tokenizer=tokenizer,
            framework="tf",
            aggregation_strategy="simple"
        )
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        st.stop()

# --- Main Application Interface ---
st.title("📖 Arabic Part-of-Speech Tagger")
input_text = st.text_area("Enter Arabic Text:", height=150, value="برلين ترفض حصول شركة امريكية على رخصة")

if st.button("Analyze Text"):
    if not input_text.strip():
        st.warning("Please enter Arabic text")
    else:
        with st.spinner("Analyzing (this may take 10-15 seconds)..."):
            try:
                classifier = load_pos_model()
                results = classifier(input_text)
                
                # Enhanced RTL display with HTML rendering
                html_output = '<div class="rtl-container">'
                for entity in results:
                    html_output += f'''
                    <div class="entity-box">
                        <b>{entity['word']}</b> - {entity['entity']}<br>
                        <small>Confidence: {entity['score']:.2f}</small>
                    </div>
                    '''
                html_output += '</div>'
                
                st.subheader("Analysis Results")
                st.markdown(html_output, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")
                st.stop()
