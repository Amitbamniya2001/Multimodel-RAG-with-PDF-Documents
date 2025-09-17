# 📄 Multimodal RAG with PDF Documents

A Streamlit app for **Retrieval-Augmented Generation (RAG)** on PDFs with both **text and images**.  
It extracts content from PDFs, chunks text, embeds text & images with **CLIP**, stores them in a vector database, and retrieves the most relevant chunks for a query. An LLM (via GitHub Models API) generates grounded answers.

---
You can try the app directly without setup:

👉 **Live Demo**: [Click here to open the app](https://multimodel-rag.streamlit.app/)

---

## 🚀 Features
- Upload and process PDFs (text + images)  
- Automatic text chunking for long documents  
- Multimodal embeddings with **CLIP (text + images)**  
- Vector search using **FAISS** (default) or **Chroma**  
- Answer generation using **ChatOpenAI** (`gpt-4.1-nano`)  
- Interactive **Streamlit UI** showing answers, text context, and retrieved images  

---
## 🤖 Models & Providers

- Embeddings: CLIP (openai/clip-vit-base-patch32)
- LLM: GitHub Models API (gpt-4.1-nano via ChatOpenAI)
- Vector Stores: FAISS (local, in-memory) or Chroma (requires sqlite ≥ 3.35)
- PDF Parsing: PyMuPDF (fitz)
- UI: Streamlit

## ⚡ Quick Start

Follow these steps to set up and run the app locally:

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/multimodal-rag.git
cd multimodal-rag
```
### 2. Create and activate a virtual environment
```bash
# On Linux / Mac
python -m venv venv
source venv/bin/activate
```
```bash
# On Windows (PowerShell)
python -m venv venv
venv\Scripts\Activate.ps1
```

### 3.Install dependencies
```bash
pip install -r requirements.txt
```
### 4. Set environment variables
```bash
GITHUB_TOKEN_CHATGPT=your_token_here
```
### 5.Run the app
```bash
streamlit run app.p
```
#### 6.Now open the URL shown in the terminal (usually http://localhost:8501) in your browser.
