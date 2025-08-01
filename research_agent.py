import PyPDF2
from typing import List
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain


# ✅ Process the uploaded PDF
def process_pdf(uploaded_file) -> List[Document]:
    reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text

    # Split into smaller chunks for embedding
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(text)

    return [Document(page_content=chunk) for chunk in chunks]


# ✅ Handle user question and get answer
def chat_with_pdf(docs, query, openai_api_key):
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectorstore = FAISS.from_documents(docs, embeddings)
    relevant_docs = vectorstore.similarity_search(query)

    llm = ChatOpenAI(temperature=0, openai_api_key=openai_api_key, model_name="gpt-4")
    chain = load_qa_chain(llm, chain_type="stuff")
    result = chain.run(input_documents=relevant_docs, question=query)
    return result
