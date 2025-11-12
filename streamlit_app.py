# streamlit_app.py

import streamlit as st
from pathlib import Path
import shutil
from src.rag.logger import get_logger
from src.rag.exceptions import RAGBaseException
from src.rag.faiss_store import FaissStore
from src.rag.rag_pipeline import RAGPipeline
from src.rag.config import settings

# ---------------- CONFIGURATION ----------------
DATA_DIR = Path(settings.data_dir)
FAISS_DIR = Path(settings.faiss_index_dir)
TOP_K = settings.top_k
LLM_MODEL = settings.llm_model_name
logger = get_logger(__name__)

# --------------- INITIALIZATION ----------------
@st.cache_resource(show_spinner=False)
def initialize_pipeline():
    """Initialize FAISS store and RAG pipeline (cached for performance)."""
    
    faiss_store = FaissStore()

    if FAISS_DIR.exists() and any(FAISS_DIR.iterdir()):
        faiss_store.load_store()
        logger.info("Loaded existing FAISS index.")
    else:
        logger.info("No FAISS index found. Creating new one...")
        # Optionally uncomment to recreate index if needed
        # from src.rag.loaders import TxtLoader
        # txt_loader = TxtLoader()
        # documents = txt_loader.load_txt(str(DATA_DIR))
        # faiss_store.create_store(documents)
        # logger.info("New FAISS index created.")

    rag_pipeline = RAGPipeline(
        vector_store=faiss_store,
        llm_model_name=LLM_MODEL
    )
    return rag_pipeline


rag_pipeline = initialize_pipeline()

# ---------------- STREAMLIT UI ----------------
st.set_page_config(page_title="Maban AI", layout="wide")

st.title("My AI Buddy")
st.markdown("Ask questions and get AI-generated answers.")

# Sidebar info
st.sidebar.header("Settings")
show_context = st.sidebar.checkbox("Show Retrieved Context", value=False)
top_k_override = st.sidebar.number_input("Top-K Retrieval", 
                                         min_value=1, max_value=10, value=TOP_K)
st.sidebar.markdown("---")
st.sidebar.info("Powered by Ollama + Llama3.2 + LangChain + FAISS")

# Query input
user_query = st.text_area("Enter your question:", placeholder="e.g., What is privacy?")
submit = st.button("Answer")

# ---------------- MAIN LOGIC ----------------
if submit and user_query.strip():
    try:
        with st.spinner("Thinking... please wait."):
            response, retrieved_docs = rag_pipeline.user_interaction(
                user_query=user_query,
                top_k=top_k_override,
                return_context=show_context
            )

        st.subheader("AI Response")
        st.markdown(response)

        if show_context:
            st.subheader("Retrieved Context")
            for i, doc in enumerate(retrieved_docs, start=1):
                st.markdown(f"**Document {i}:**\n{doc.page_content}")
                if doc.metadata:
                    st.caption(str(doc.metadata))
                st.markdown("---")

    except RAGBaseException as e:
        st.error(f"RAG Error: {str(e)}")
        logger.error(f"RAGBaseException: {e}")
    except Exception as e:
        st.exception(f"Unexpected Error: {e}")
        logger.exception(f"Unexpected error: {e}")

else:
    st.info("Enter a question and click 'Answer'.")

