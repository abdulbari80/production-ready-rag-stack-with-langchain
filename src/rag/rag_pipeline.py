from typing import List, Optional
from langchain_core.documents import Document

from src.rag.logger import get_logger
from src.rag.exceptions import VectorStoreNotInitializedError
from src.rag.faiss_store import FaissStore
from src.rag.llm_connector import get_llm_response

logger = get_logger(__name__)


class RAGPipeline:
    """
    Retrieval-Augmented Generation (RAG) pipeline.
    Combines a vector store for retrieval with an LLM for response generation.
    """

    def __init__(
        self,
        vector_store: Optional[FaissStore] = None,
        top_k: int = 3,
        llm_model_name: Optional[str] = None,
    ):
        """
        Initialize RAGPipeline.

        Args:
            vector_store (Optional[FaissStore]): Pre-initialized FAISS store instance.
            top_k (int): Number of top documents to retrieve for context.
            llm_model_name (Optional[str]): Optional LLM model name for response generation.
        """
        self.vector_store = vector_store or FaissStore()
        self.top_k = top_k
        self.llm_model_name = llm_model_name
        logger.info(f"RAGPipeline initialized with top_k={self.top_k}")

    def add_documents(self, documents: List[Document]):
        """
        Add documents to the vector store.

        Args:
            documents (List[Document]): List of documents to add.
        """
        if not documents:
            logger.warning("No documents provided to add to the vector store.")
            return

        if self.vector_store.is_loaded:
            # Merge new docs into existing store
            logger.info(f"Adding {len(documents)} documents to existing FAISS store...")
            try:
                self.vector_store.vector_store.add_documents(documents)
                self.vector_store.vector_store.save_local(self.vector_store.store_dir)
                logger.info("Documents successfully added and FAISS index updated.")
            except Exception as e:
                logger.exception("Failed to add documents to FAISS store.")
                raise
        else:
            # Create new store
            logger.info("FAISS store not initialized. Creating new store...")
            self.vector_store.create_store(documents)

    def retrieve(self, query: str) -> List[Document]:
        """
        Retrieve top-k most similar documents from the vector store.

        Args:
            query (str): Query string to search.

        Returns:
            List[Document]: List of retrieved documents.
        """
        if not self.vector_store.is_loaded:
            logger.error("Vector store is not initialized for retrieval.")
            raise VectorStoreNotInitializedError()

        logger.info(f"Retrieving top {self.top_k} documents for query: {query}")
        return self.vector_store.similarity_search(query, k=self.top_k)

    def generate_answer(self, query: str, context_docs: List[Document]) -> str:
        """
        Generate an answer using the LLM with retrieved context.

        Args:
            query (str): User query.
            context_docs (List[Document]): List of documents retrieved as context.

        Returns:
            str: LLM-generated response.
        """
        context_text = "\n".join([doc.page_content for doc in context_docs])
        logger.info("Generating response using LLM with retrieved context...")
        response = get_llm_response(prompt=query, model_name=self.llm_model_name)
        logger.info("Response generation complete.")
        return response

    def query(self, user_query: str) -> str:
        """
        End-to-end RAG query:
        1. Retrieve relevant documents
        2. Generate answer from LLM

        Args:
            user_query (str): User input query.

        Returns:
            str: Generated answer.
        """
        logger.info(f"Processing query through RAGPipeline: {user_query}")
        retrieved_docs = self.retrieve(user_query)
        answer = self.generate_answer(user_query, retrieved_docs)
        return answer
