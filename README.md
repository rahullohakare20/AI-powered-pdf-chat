# 📄 Chat with Your PDF – AI-Powered PDF Q&A App

This is an open-source Streamlit app that allows you to upload a PDF and ask natural language questions about its content. Powered by Hugging Face transformers and vector search using FAISS.

---

## 🚀 Features

- Upload and process any PDF file
- Semantic search using HuggingFace Embeddings (`all-MiniLM-L6-v2`)
- Q&A with context-aware answers using `distilbert-base-cased-distilled-squad`
- Clean UI with custom CSS
- Modular codebase

---

## 🧠 How It Works

1. **PDF Upload:** Extracts text using `PyPDF2`.
2. **Text Chunking:** Splits the content into manageable chunks.
3. **Embeddings:** Converts text into vector space using sentence transformers.
4. **Vector Search:** Finds relevant chunks using FAISS.
5. **Answer Generation:** Feeds question and context into a transformer model to get an answer.

---

## 🛠️ Installation

```bash
git clone https://github.com/your-username/chat-with-pdf.git
cd chat-with-pdf
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## ▶️ Running the App

```bash
streamlit run app.py
```

---

## 📁 Project Structure

```
chat-with-pdf/
├── app.py               # Main Streamlit UI logic
├── pdf_utils.py         # PDF reading utilities
├── qa_pipeline.py       # Embeddings & Q&A pipeline
├── style.css            # Custom CSS
├── requirements.txt     # Python dependencies
├── .gitignore           # Files to ignore in Git
└── README.md            # You're here!
```

---

## ✅ Example Question

Upload a PDF with contract terms and try:

> "What is the agreement date?"

Or upload an academic paper and ask:

> "What is the main conclusion of the study?"

---

## 📦 Dependencies

- `streamlit`
- `PyPDF2`
- `transformers`
- `langchain`
- `faiss-cpu`
- `sentence-transformers`

---

## 📄 License

MIT License. Feel free to modify and use.