import os
from pathlib import Path
import shutil
from src.rag.logger import get_logger
from src.rag.exceptions import RAGBaseException
from src.rag.faiss_store import FaissStore
from src.rag.rag_pipeline import RAGPipeline
from src.rag.loaders import TxtLoader
from src.rag.config import settings

# ---------------- CONFIGURATION ----------------
DATA_DIR = Path(settings.data_dir)  # e.g., "data/privacy"
FAISS_DIR = Path(settings.faiss_index_dir)  # e.g., "vector_store_au/faiss_index"
TOP_K = settings.top_k
LLM_MODEL = settings.llm_model_name
logger = get_logger(__name__)


# ---------------- MAIN PIPELINE ----------------
def main():
    """Main entry point for the local RAG pipeline execution."""
    try:
        logger.info("Starting RAG pipeline execution >>>>>")

        # # Step 1: Load and preprocess TXT documents
        # logger.info("Step 1: Loading and chunking TXT documents...")
        # txt_loader = TxtLoader()
        # logger.info(f"Loading and chunking TXT from: {DATA_DIR}")
        # documents = txt_loader.load_txt(str(DATA_DIR))

        # if not documents:
        #     logger.warning(f"No documents found in directory: {DATA_DIR}")
        #     return

        # logger.info(f"Loaded {len(documents)} document chunks.")
        # logger.info(">>>>>> Text documents loaded and chunked successfully.")

        # # Step 2: Initialize FAISS Vector Store
        # if FAISS_DIR.exists() and any(FAISS_DIR.iterdir()):
        #     shutil.rmtree(FAISS_DIR, ignore_errors=True)  # Clear existing index for fresh start
        #     logger.info("Old FaissStore removed.")
        faiss_store = FaissStore()

        if FAISS_DIR.exists() and any(FAISS_DIR.iterdir()):
            faiss_store.load_store()
        else:
            logger.info("No FAISS index found. Creating new index from documents...")
            # faiss_store.create_store(documents)

        # Step 3: Initialize RAG pipeline
        vector_store = faiss_store
        rag_pipeline = RAGPipeline(vector_store=vector_store, 
                                   top_k=TOP_K, llm_model_name=LLM_MODEL)

        # Step 4: Execute query through RAG pipeline
        continued = 'y'
        while continued == 'y':
            response = rag_pipeline.user_interaction()
            logger.info("=" * 60)
            logger.info(f"RAG Response: {response}")
            logger.info("=" * 60 + "\n")
            continued = str(input("\n\nDo you want to continue [y/n]: "))[0].lower()
            if continued != 'y':
                logger.info("Thanks for using the RAG pipeline. Exiting now.")
                break

    except RAGBaseException as e:
        logger.error(f"RAG-specific error occurred: {str(e)}")
    except Exception as e:
        logger.exception(f"Unexpected error occurred: {str(e)}")
    finally:
        logger.info("\nRAG pipeline execution completed.")


if __name__ == "__main__":
    main()
