import pdfplumber
import os
from dotenv import load_dotenv
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI

# Load .env
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Set OpenAI API key
os.environ["OPENAI_API_KEY"] = openai_api_key

def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        return "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())

def split_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    return splitter.split_text(text)

def embed_texts(chunks):
    embeddings = OpenAIEmbeddings()
    return FAISS.from_texts(chunks, embeddings)

def query_vectorstore(query, db):
    docs = db.similarity_search(query, k=5)
    llm = OpenAI(temperature=0)
    chain = load_qa_chain(llm, chain_type="stuff")
    return chain.run(input_documents=docs, question=query)

def run_agent(pdf_path, question):
    text = extract_text_from_pdf(pdf_path)
    chunks = split_text(text)
    db = embed_texts(chunks)
    return query_vectorstore(question, db)
