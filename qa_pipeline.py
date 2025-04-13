from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from transformers import pipeline

def create_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_texts(chunks, embedding=embeddings)

def load_qa_pipeline():
    return pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

def ask_question(qa_pipeline, query, vectorstore):
    docs = vectorstore.similarity_search(query, k=1)
    context = docs[0].page_content.strip() if docs and docs[0].page_content else ""
    return qa_pipeline({"question": query, "context": context}), context
