import os
from typing import List, Optional

from langchain_community.vectorstores import FAISS
# from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

# Custom modules
from src.rag.logger import get_logger
from src.rag.exceptions import (
    VectorStoreNotInitializedError,
    DocumentLoadError,
    EmbeddingModelError,
    RetrievalError)

from .config import settings

logger = get_logger(__name__)

MODEL_NAME = settings.embedding_model_name
DEFAULT_STORE_DIR = settings.faiss_index_dir


class FaissStore:
    """
    Wrapper around LangChain's FAISS vector store.
    Handles index creation, saving, loading, and similarity search with robust logging,
    dependency injection, and error handling.
    """

    def __init__(
        self,
        embedding_model: Optional[HuggingFaceEmbeddings] = None):
        """
        Initialize FAISS store wrapper.

        Args:
            embedding_model (Optional[SentenceTransformerEmbeddings]): Optional custom embedding model.
        """
        self.embedding_model = embedding_model or HuggingFaceEmbeddings(model_name=MODEL_NAME)
        self.store_dir = DEFAULT_STORE_DIR
        self.vector_store: Optional[FAISS] = None

        # Ensure store directory exists
        os.makedirs(self.store_dir, exist_ok=True)

    @property
    def is_loaded(self) -> bool:
        """Check if vector store has been initialized or loaded."""
        return self.vector_store is not None

    def create_store(self, documents: List[Document]) -> FAISS:
        """
        Create and persist a FAISS vector index from a list of documents.

        Args:
            documents (List[Document]): Documents to embed and store.

        Returns:
            FAISS: FAISS vector store.

        Raises:
            EmbeddingModelError: If embedding fails.
            DocumentLoadError: If saving the index fails.
        """
        if not documents:
            raise ValueError("No documents provided for FAISS index creation.")

        try:
            logger.info(f"Creating FAISS store with {len(documents)} documents...")
            self.vector_store = FAISS.from_documents(documents, 
                                                     embedding=self.embedding_model,
                                                     normalize_L2 = True
                                                     )
            self.vector_store.save_local(self.store_dir)
            logger.info(
                f"FAISS store successfully created and saved at: {self.store_dir} "
                f"with {self.vector_store.index.ntotal} vectors.")
            
            return self.vector_store

        except Exception as e:
            logger.exception("Error while creating FAISS store.")
            raise EmbeddingModelError(f"Failed to create FAISS index: {str(e)}") from e

    def load_store(self) -> FAISS:
        """
        Load an existing FAISS index from disk.

        Returns:
            FAISS: The loaded FAISS vector store.

        Raises:
            DocumentLoadError: If the index cannot be found or loaded.
        """
        try:
            self.vector_store = FAISS.load_local(
                self.store_dir,
                embeddings=self.embedding_model,
                allow_dangerous_deserialization=True
            )
            return self.vector_store

        except FileNotFoundError as e:
            logger.error(f"FAISS store not found at {self.store_dir}.")
            raise DocumentLoadError(f"FAISS index not found at path: {self.store_dir}") from e
        except Exception as e:
            logger.exception("Error while loading FAISS store.")
            raise DocumentLoadError(f"Failed to load FAISS store: {str(e)}") from e

    def similarity_search(self, query: str="What is privacy?", 
                          k: int = 3) -> List[Document]:
        """
        Perform a similarity search on the FAISS index.

        Args:
            query (str): Query string to search for.
            k (int): Number of top results to retrieve.

        Returns:
            List[Document]: List of most similar documents.

        Raises:
            VectorStoreNotInitializedError: If the store hasn't been created or loaded.
            RetrievalError: If the search fails.
        """
        if not self.is_loaded:
            logger.error("Attempted similarity search before initializing FAISS store.")
            raise VectorStoreNotInitializedError()

        try:
            results = self.vector_store.similarity_search_with_score(query, k=k)
            return results

        except Exception as e:
            logger.exception("Error during similarity search.")
            raise RetrievalError(f"Failed similarity search for query '{query}': {str(e)}") from e
