import streamlit as st
from pdf_utils import extract_text_from_pdf, split_text
from qa_pipeline import create_vectorstore, load_qa_pipeline, ask_question

st.set_page_config(page_title="Chat with PDF", layout="wide")

# Load custom CSS
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title("💬 AI-Powered PDF Chat")

pdf = st.file_uploader("📄 Upload your PDF", type="pdf")

if pdf:
    text = extract_text_from_pdf(pdf)
    chunks = split_text(text)
    
    st.success(f"✅ Extracted and split {len(chunks)} chunks from PDF.")

    vectorstore = create_vectorstore(chunks)
    qa_pipeline = load_qa_pipeline()

    query = st.text_input("❓ Ask a question about your PDF:")
    if query:
        result, context = ask_question(qa_pipeline, query, vectorstore)

        # st.subheader("📚 Context Sent to Model:")
        # st.write(context or "No relevant context found.")

        st.subheader("📄 Answer:")
        st.write(result["answer"])
