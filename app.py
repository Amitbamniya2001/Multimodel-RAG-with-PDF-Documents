import streamlit as st
import fitz  # PyMuPDF
from langchain_core.documents import Document
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import numpy as np
from langchain_openai import ChatOpenAI
from langchain.schema.messages import HumanMessage
import os
import base64
import io
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

# from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores import DocArrayInMemorySearch

from langchain.embeddings.base import Embeddings

# --- 1. SET UP THE ENVIRONMENT & MODELS (RUNS ONCE) ---

# Load environment variables from .env file
load_dotenv()

st.set_page_config(page_title="Multimodal PDF RAG", layout="wide")
st.title("📄 Multimodal RAG with PDF Documents")

@st.cache_resource
def load_models():
    """Load and cache the CLIP model, processor, and LLM."""
    # Load the CLIP model for embeddings
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model.eval()

    # Load the LLM using GitHub's endpoint
    token = os.getenv("GITHUB_TOKEN_CHATGPT")
    if not token:
        st.error("GitHub token not found! Please set it in secrets or .env.")
        st.stop()
        
    endpoint = "https://models.github.ai/inference"
    llm = ChatOpenAI(
        model="gpt-4.1-nano",
        openai_api_key=token,
        openai_api_base=endpoint,
        model_kwargs={"max_tokens": 1024}
    )
    return clip_model, clip_processor, llm

clip_model, clip_processor, llm = load_models()

# --- 2. EMBEDDING FUNCTIONS ---

def embed_image(image_data):
    """Embed image using CLIP."""
    image = image_data.convert("RGB")
    inputs = clip_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        features = clip_model.get_image_features(**inputs)
        features /= features.norm(dim=-1, keepdim=True)
        return features.squeeze().numpy()

def embed_text(text):
    """Embed text using CLIP."""
    inputs = clip_processor(
        text=text, return_tensors="pt", padding=True, truncation=True, max_length=77
    )
    with torch.no_grad():
        features = clip_model.get_text_features(**inputs)
        features /= features.norm(dim=-1, keepdim=True)
        return features.squeeze().numpy()

# Wrap CLIP into a LangChain-compatible embedding class
class ClipEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [embed_text(t) for t in texts]

    def embed_query(self, text):
        return embed_text(text)

clip_embeddings = ClipEmbeddings()

# --- 3. PDF PROCESSING ---

@st.cache_resource(show_spinner="Processing PDF... This may take a moment.")
def process_pdf(uploaded_file):
    """Process the uploaded PDF to extract text and images, and create embeddings."""
    if uploaded_file is None:
        return None, None

    file_bytes = uploaded_file.read()
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    
    all_docs, image_data_store = [], {}
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

    for i, page in enumerate(doc):
        # Process text
        text = page.get_text()
        if text.strip():
            temp_doc = Document(page_content=text, metadata={"page": i, "type": "text"})
            text_chunks = splitter.split_documents([temp_doc])
            for chunk in text_chunks:
                all_docs.append(chunk)

        # Process images
        for img_index, img in enumerate(page.get_images(full=True)):
            try:
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                
                image_id = f"page_{i}_img_{img_index}"
                
                buffered = io.BytesIO()
                pil_image.save(buffered, format="PNG")
                img_base64 = base64.b64encode(buffered.getvalue()).decode()
                image_data_store[image_id] = img_base64
                
                image_doc = Document(
                    page_content=f"[Image: {image_id}]",
                    metadata={"page": i, "type": "image", "image_id": image_id}
                )
                all_docs.append(image_doc)
            except Exception as e:
                st.error(f"Error processing image {img_index} on page {i}: {e}")
                continue
    doc.close()

    if not all_docs:
        return None, None
        
    # Build Chroma vector store
    texts = [d.page_content for d in all_docs]
    metadatas = [d.metadata for d in all_docs]

    # vector_store = Chroma.from_texts(
    #     texts=texts,
    #     embedding=clip_embeddings,
    #     metadatas=metadatas
    # )
    vector_store = DocArrayInMemorySearch.from_texts(
    texts=texts,
    embedding=clip_embeddings,
    metadatas=metadatas
    )   
    return vector_store, image_data_store

# --- 4. RAG PIPELINE FUNCTIONS ---

def retrieve_multimodal(query, vector_store, k=5):
    results = vector_store.similarity_search(query, k=k)
    return results

def create_multimodal_message(query, retrieved_docs, image_data_store):
    content = [{"type": "text", "text": f"Question: {query}\n\nContext:\n"}]
    text_docs = [doc for doc in retrieved_docs if doc.metadata.get("type") == "text"]
    image_docs = [doc for doc in retrieved_docs if doc.metadata.get("type") == "image"]
    
    if text_docs:
        text_context = "\n\n".join([f"[Page {doc.metadata['page']}]: {doc.page_content}" for doc in text_docs])
        content.append({"type": "text", "text": f"Text excerpts:\n{text_context}\n"})
    
    for doc in image_docs:
        image_id = doc.metadata.get("image_id")
        if image_id and image_id in image_data_store:
            content.append({"type": "text", "text": f"\n[Image from page {doc.metadata['page']}]:\n"})
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{image_data_store[image_id]}"}
            })
    content.append({"type": "text", "text": "\n\nPlease answer the question based on the provided text and images."})
    return HumanMessage(content=content)

# --- 5. STREAMLIT UI ---

if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'image_data_store' not in st.session_state:
    st.session_state.image_data_store = None
if 'processed' not in st.session_state:
    st.session_state.processed = False

with st.sidebar:
    st.header("Setup")
    uploaded_file = st.file_uploader("Upload your PDF file", type="pdf")

    if uploaded_file:
        if st.button("Process Document"):
            st.session_state.vector_store, st.session_state.image_data_store = process_pdf(uploaded_file)
            if st.session_state.vector_store:
                st.success("Document processed successfully!")
                st.session_state.processed = True
            else:
                st.error("Could not process the document. Make sure it contains text or images.")
                st.session_state.processed = False

st.divider()

if not st.session_state.processed:
    st.info("Please upload a PDF and click 'Process Document' to begin.")
else:
    st.success("Document is ready. You can now ask questions.")
    query = st.text_input("Ask a question about the document:", key="query_input")
    
    if st.button("Get Answer"):
        if not query:
            st.warning("Please enter a question.")
        else:
            with st.spinner("Retrieving context and generating answer..."):
                context_docs = retrieve_multimodal(query, st.session_state.vector_store)
                message = create_multimodal_message(query, context_docs, st.session_state.image_data_store)
                response = llm.invoke([message])
                answer = response.content

                st.subheader("Answer")
                st.markdown(answer)
                st.divider()
                
                st.subheader("Retrieved Context")
                retrieved_text = [doc for doc in context_docs if doc.metadata.get("type") == "text"]
                retrieved_images = [doc for doc in context_docs if doc.metadata.get("type") == "image"]
                
                if retrieved_text:
                    with st.expander("Show Retrieved Text", expanded=True):
                        for doc in retrieved_text:
                            st.info(f"**From Page {doc.metadata.get('page', '?')}:**")
                            st.text(doc.page_content)

                if retrieved_images:
                     with st.expander("Show Retrieved Images", expanded=True):
                        cols = st.columns(len(retrieved_images))
                        for i, doc in enumerate(retrieved_images):
                            image_id = doc.metadata.get("image_id")
                            if image_id and image_id in st.session_state.image_data_store:
                                with cols[i]:
                                    st.image(
                                        io.BytesIO(base64.b64decode(st.session_state.image_data_store[image_id])),
                                        caption=f"Image from Page {doc.metadata.get('page', '?')}",
                                        use_column_width=True
                                    )
