import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from transformers import pipeline

# Set page config
st.set_page_config(page_title="AI PDF Chat", page_icon="📄", layout="wide")

# Custom CSS for beautiful UI
st.markdown("""
    <style>
        .main {
            background-color: #f9f9f9;
        }
        .stTextInput > div > div > input {
            padding: 0.75rem;
            border-radius: 0.5rem;
            border: 1px solid #ccc;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            padding: 0.6rem 1.2rem;
            border: none;
            border-radius: 0.5rem;
            font-size: 1rem;
        }
        .stMarkdown h1 {
            font-size: 2.5rem;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

# Title & Header
st.title("💬 AI-Powered PDF Chat")
st.markdown("Ask questions directly from your uploaded PDF using powerful AI models. ⚡")

# Upload Section
with st.container():
    pdf = st.file_uploader("📎 Upload your PDF file", type="pdf")

# PDF Processing
if pdf is not None:
    with st.spinner("📖 Reading and analyzing your PDF..."):
        reader = PdfReader(pdf)
        text = ""
        for page in reader.pages:
            text += page.extract_text()

        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        texts = text_splitter.split_text(text)

        # Create vectorstore
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_texts(texts, embedding=embeddings)

        # Load Q&A pipeline
        qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

    # Question input
    st.subheader("🔍 Ask a Question")
    query = st.text_input("Type your question below and hit Enter")

    if query:
        with st.spinner("🤖 Finding the best answer..."):
            docs = vectorstore.similarity_search(query, k=1)
            context = docs[0].page_content.strip() if docs and docs[0].page_content else ""

            st.markdown("#### 📘 Context Used")
            st.code(context)

            if context:
                inputs = {"question": query, "context": context}
                result = qa_pipeline(inputs)
                answer = result["answer"]
            else:
                answer = "❌ No relevant context found."

            st.markdown("#### ✅ Answer")
            st.success(answer)
