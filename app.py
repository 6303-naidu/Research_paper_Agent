import os
import streamlit as st
from research_agent import process_pdf, chat_with_pdf

# Load OpenAI API key from environment variable
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Check if API key is set
if not OPENAI_API_KEY:
    st.error("❌ OPENAI_API_KEY not found. Please set it in your environment variables.")
    st.stop()

# Set Streamlit page config
st.set_page_config(page_title="🧠 AI Research Agent", layout="wide")

# App title
st.title("🧠 AI Research Agent")
st.write("Upload a research paper (PDF) and ask questions about it using GPT-4.")

# Upload PDF
uploaded_file = st.file_uploader("📄 Upload your research PDF", type=["pdf"])

# Show input box and result only if PDF is uploaded
if uploaded_file:
    # Process the PDF (extract text and chunk it)
    with st.spinner("🔍 Reading and chunking your PDF..."):
        docs = process_pdf(uploaded_file)

    st.success("✅ PDF processed successfully!")

    # Chat interface
    user_question = st.text_input("💬 Ask a question about the paper:")

    if user_question:
        with st.spinner("🤖 Thinking..."):
            result = chat_with_pdf(docs, user_question, OPENAI_API_KEY)
        st.markdown("### 🧠 Answer:")
        st.write(result)
