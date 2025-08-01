import os
import streamlit as st
from research_agent import process_pdf, chat_with_pdf

st.set_page_config(page_title="🧠 AI Research Agent", layout="wide")
st.title("🧠 AI Research Agent")
st.write("Upload a PDF and ask questions using GPT-4.")

# Load API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("❌ OPENAI_API_KEY not found. Set it in Render or your .env.")
    st.stop()

# Upload PDF
uploaded_file = st.file_uploader("📄 Upload a PDF file", type=["pdf"])
if uploaded_file:
    with st.spinner("📚 Reading PDF..."):
        docs = process_pdf(uploaded_file)
    st.success("✅ PDF processed successfully.")

    user_query = st.text_input("💬 Ask a question:")
    if user_query:
        with st.spinner("🤖 Thinking..."):
            answer = chat_with_pdf(docs, user_query, OPENAI_API_KEY)
        st.markdown("### ✅ Answer")
        st.write(answer)
