# Import libraries
import streamlit as st
from transformers import pipeline

# Configure Streamlit page
st.set_page_config(
    page_title="Arabic POS Tagger",
    page_icon="📚",
    layout="wide"
)

# Custom CSS for RTL support and styling
st.markdown("""
<style>
    .rtl-text {
        direction: rtl;
        text-align: right;
        font-size: 18px;
    }
    .tag-box {
        border: 1px solid #4a4a4a;
        border-radius: 5px;
        padding: 5px;
        margin: 2px;
        display: inline-block;
        background-color: #f0f2f6;
    }
</style>
""", unsafe_allow_html=True)

# Load model with caching
@st.cache_resource(show_spinner="Loading POS model...")
def load_pos_model():
    return pipeline(
        "ner",
        model="MariamOsama3/Mariam_classifer2",
        aggregation_strategy="simple"
    )

# App header
st.title("📖 Arabic Part-of-Speech Tagger")
st.markdown("Analyze Arabic text with POS tagging using transformer models")

# Input section
with st.container():
    input_text = st.text_area(
        "Enter Arabic Text:",
        height=150,
        value="برلين ترفض حصول شركة امريكية على رخصة",
        key="input_text"
    )

# Processing section
if st.button("Analyze Text", type="primary"):
    if not input_text.strip():
        st.warning("Please enter some Arabic text")
    else:
        with st.spinner("Analyzing text..."):
            pos_model = load_pos_model()
            results = pos_model(input_text)
            
            # Display results in two columns
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Detailed Analysis")
                st.markdown('<div class="rtl-text">', unsafe_allow_html=True)
                for entity in results:
                    st.markdown(
                        f'<div class="tag-box">{entity["word"]} '
                        f'<span style="color: #ff4b4b;">({entity["entity"]})</span></div>',
                        unsafe_allow_html=True
                    )
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.subheader("POS Tag Summary")
                
                # Create tag frequency chart
                tag_counts = {}
                for entity in results:
                    tag = entity["entity"]
                    tag_counts[tag] = tag_counts.get(tag, 0) + 1
                
                if tag_counts:
                    st.bar_chart(tag_counts)
                else:
                    st.info("No POS tags found")

                # Display raw JSON output
                with st.expander("Show Raw Output"):
                    st.json(results)

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using [Streamlit](https://streamlit.io) & [Hugging Face](https://huggingface.co)")
