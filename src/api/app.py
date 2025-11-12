from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
import shutil
import uvicorn

from src.rag.logger import get_logger
from src.rag.exceptions import RAGBaseException
from src.rag.faiss_store import FaissStore
from src.rag.rag_pipeline import RAGPipeline
from src.rag.loaders import PDFLoader
from src.rag.config import settings


# ---------------- CONFIGURATION ----------------
DATA_DIR = Path(settings.data_dir)           # e.g., "data/resumes"
FAISS_DIR = Path(settings.faiss_index_dir)   # e.g., "vector_store/faiss_index"
TOP_K = settings.top_k
LLM_MODEL = settings.llm_model_name or "gemma:2b"
logger = get_logger(__name__)

app = FastAPI(title="RAG Application API", version="1.0")

# Global RAG pipeline instance
rag_pipeline = None


# ---------------- REQUEST MODEL ----------------
class QueryRequest(BaseModel):
    query: str


# ---------------- STARTUP EVENT ----------------
@app.on_event("startup")
def load_rag_pipeline():
    """
    Initialize and load the RAG pipeline on startup so it’s ready for incoming requests.
    """
    global rag_pipeline

    try:
        logger.info("Starting RAG pipeline initialization...")

        # Step 1: Load PDF documents
        pdf_loader = PDFLoader()
        documents = pdf_loader.load_pdf(str(DATA_DIR))

        if not documents:
            logger.warning(f"No documents found in {DATA_DIR}")
            return

        logger.info(f"Loaded {len(documents)} document chunks.")

        # Step 2: Prepare FAISS store
        if not FAISS_DIR.exists() or not any(FAISS_DIR.iterdir()):
            logger.info("No existing FAISS index found — creating a new one.")
            faiss_store = FaissStore(store_dir=str(FAISS_DIR))
            faiss_store.create_store(documents)
        else:
            logger.info("Loading existing FAISS index.")
            faiss_store = FaissStore(store_dir=str(FAISS_DIR))
            faiss_store.load_store()

        # Step 3: Initialize RAG pipeline
        rag_pipeline = RAGPipeline(
            vector_store=faiss_store,
            top_k=TOP_K,
            llm_model_name=LLM_MODEL
        )

        logger.info("RAG pipeline initialized successfully.")

    except Exception as e:
        logger.exception(f"Failed to initialize RAG pipeline: {str(e)}")
        raise e


# ---------------- ROUTES ----------------
@app.post("/query")
async def run_rag_query(request: QueryRequest):
    """
    Endpoint to query the RAG pipeline.
    """
    global rag_pipeline

    if not rag_pipeline:
        raise HTTPException(status_code=500, detail="RAG pipeline not initialized yet.")

    try:
        logger.info(f"Received user query: {request.query}")
        response = rag_pipeline.query(request.query)
        return {"query": request.query, "response": response}
    except RAGBaseException as e:
        logger.error(f"RAG-specific error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


# ---------------- RUN LOCALLY ----------------
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
