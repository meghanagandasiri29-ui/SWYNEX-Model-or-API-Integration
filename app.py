
import streamlit as st
from transformers import pipeline

st.set_page_config(
    page_title="SWYNEX AI Text Summarizer",
    page_icon="📝"
)

st.title("📝 AI Text Summarizer")
st.write("Summarize long text using a locally running AI model.")

@st.cache_resource
def load_model():
    return pipeline(
        "summarization",
        model="sshleifer/distilbart-cnn-12-6"
    )

text = st.text_area(
    "Enter the text you want to summarize:",
    height=250,
    placeholder="Paste an article or paragraph here..."
)

if st.button("Summarize Text"):
    if not text.strip():
        st.warning("Please enter some text first.")
    elif len(text.split()) < 30:
        st.warning("Please enter a longer text (at least 30 words).")
    else:
        try:
            with st.spinner("Loading AI model and summarizing..."):
                summarizer = load_model()
                result = summarizer(
                    text,
                    max_length=100,
                    min_length=20,
                    do_sample=False,
                    truncation=True
                )

            st.subheader("Generated Summary")
            st.write(result[0]["summary_text"])

        except Exception as error:
            st.error("Sorry, the model could not summarize this text.")
            st.code(str(error))

st.divider()
st.caption("Powered by a local Transformers AI model")
