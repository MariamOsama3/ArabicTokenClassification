# Replace the environment config at the top with:
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_KERAS"] = "1"

# Modify your model loading function like this:
@st.cache_resource(show_spinner="Loading POS model...")
def load_pos_model():
    # Force TensorFlow backend for Keras
    import tensorflow as tf
    tf.keras.utils.set_keras_mod_submodules(
        backend_module="tensorflow.keras",
        layers_module="tensorflow.keras.layers",
        models_module="tensorflow.keras.models",
        utils_module="tensorflow.keras.utils"
    )
    
    # Load model with explicit Keras config
    model = TFAutoModelForTokenClassification.from_pretrained(
        "MariamOsama3/Mariam_classifer2",
        from_pt=True  # Add this if you converted from PyTorch
    )
    tokenizer = AutoTokenizer.from_pretrained("MariamOsama3/Mariam_classifer2")
    
    return pipeline(
        "ner",
        model=model,
        tokenizer=tokenizer,
        framework="tf",
        aggregation_strategy="simple"
    )
