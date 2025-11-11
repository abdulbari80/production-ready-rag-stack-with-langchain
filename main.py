import os
from pathlib import Path
import shutil
from src.rag.logger import get_logger
from src.rag.exceptions import RAGBaseException
from src.rag.faiss_store import FaissStore
from src.rag.rag_pipeline import RAGPipeline
from src.rag.loaders import PDFLoader
from src.rag.config import settings

# ---------------- CONFIGURATION ----------------
DATA_DIR = Path(settings.data_dir)  # e.g., "data/resumes"
FAISS_DIR = Path(settings.faiss_index_dir)  # e.g., "vector_store/faiss_index"
TOP_K = settings.top_k
LLM_MODEL = settings.llm_model_name or "gemma:2b"  # fallback model
logger = get_logger(__name__)


# ---------------- MAIN PIPELINE ----------------
def main():
    """Main entry point for the local RAG pipeline execution."""
    try:
        logger.info("Starting RAG pipeline execution")

        # Step 1: Load and preprocess PDF documents
        pdf_loader = PDFLoader()
        logger.info(f"Loading and chunking PDFs from: {DATA_DIR}")
        documents = pdf_loader.load_pdf(str(DATA_DIR))

        if not documents:
            logger.warning(f"No documents found in directory: {DATA_DIR}")
            return

        logger.info(f"Loaded {len(documents)} document chunks.")

        # Step 2: Initialize FAISS Vector Store
        if FAISS_DIR.exists() and any(FAISS_DIR.iterdir()):
            shutil.rmtree("vector_store/faiss_index", ignore_errors=True)  # Clear existing index for fresh start
        faiss_store = FaissStore(store_dir=str(FAISS_DIR))

        if FAISS_DIR.exists() and any(FAISS_DIR.iterdir()):
            logger.info("Existing FAISS index detected. Loading index...")
            faiss_store.load_store()
        else:
            logger.info("No FAISS index found. Creating new index from documents...")
            faiss_store.create_store(documents)

        # Step 3: Initialize RAG pipeline
        rag_pipeline = RAGPipeline(
            vector_store=faiss_store,
            top_k=TOP_K,
            llm_model_name=LLM_MODEL
        )

        # Step 4: Accept user query (for testing, hardcoded)
        user_query = "Summarize my professional experience and key skills."
        logger.info(f"Running RAG query: {user_query}")

        # Step 5: Execute query through RAG pipeline
        response = rag_pipeline.query(user_query)

        # Step 6: Display response
        print("\n" + "=" * 60)
        print("RAG Response:")
        print(response)
        print("=" * 60 + "\n")

    except RAGBaseException as e:
        logger.error(f"RAG-specific error occurred: {str(e)}")
    except Exception as e:
        logger.exception(f"Unexpected error occurred: {str(e)}")
    finally:
        logger.info("RAG pipeline execution completed.")


if __name__ == "__main__":
    main()
