# Remove this line from the top imports:
# import tensorflow as tf  # Only if you're not using direct TF operations

# Modify your model loading function:
@st.cache_resource(show_spinner="Loading POS model...")
def load_pos_model():
    # Force TensorFlow compatibility
    from tensorflow import keras
    import tensorflow as tf
    
    # Load model WITHOUT PyTorch conversion
    model = TFAutoModelForTokenClassification.from_pretrained(
        "MariamOsama3/Mariam_classifer2",
        from_pt=False  # Explicitly disable PyTorch conversion
    )
    tokenizer = AutoTokenizer.from_pretrained("MariamOsama3/Mariam_classifer2")
    
    return pipeline(
        "ner",
        model=model,
        tokenizer=tokenizer,
        framework="tf",
        aggregation_strategy="simple"
    )
