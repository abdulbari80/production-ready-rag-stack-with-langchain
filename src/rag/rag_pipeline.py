from typing import List, Optional
from langchain_core.documents import Document

from src.rag.logger import get_logger
from src.rag.exceptions import VectorStoreNotInitializedError
from src.rag.faiss_store import FaissStore
from src.rag.config import settings 

from langchain_ollama import ChatOllama  # or OllamaLLM
from langchain_core.messages import SystemMessage, HumanMessage

MODEL_NAME = settings.llm_model_name
TEMPERATURE = settings.llm_temperature
THRESHOLD = settings.threshold

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
        llm_model_name: Optional[str] = None):
        """
        Initialize RAGPipeline.

        Args:
            vector_store (Optional[FaissStore]): Pre-initialized FAISS store instance.
            top_k (int): Number of top documents to retrieve for context.
            llm_model_name (Optional[str]): Optional LLM model name for response generation.
        """
        self.vector_store = vector_store or FaissStore()
        self.top_k = top_k
        self.llm_model_name = llm_model_name or MODEL_NAME

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

    def filter_retrieved_context(self, query: str,
                                 threshold: float=THRESHOLD) -> List[Document]:
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

        sim_results = self.vector_store.similarity_search(query, k=self.top_k) 
        filtered_context_docs = [doc for doc, score in sim_results if score > threshold]
        return filtered_context_docs

    def generate_response(self, query: str, context_docs: List[Document]) -> str:
        """
        Generate a response using the LLM with retrieved context.

        Args:
            query (str): User query.
            context_docs (List[Document]): List of documents retrieved as context.

        Returns:
            str: LLM-generated response.
        """
        if context_docs:
            retrieved_context = "\n\n".join([doc.page_content for doc in context_docs])
            system_prompt = """
            You're an expert assitant. This application is desired to respond to
            the user queries on Australian Privacy laws, regulations, frameworks,
            policies and best practices. Please the context to answer the question.
            """

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"{retrieved_context}\n\nQuestion: {query}")
            ]
            llm = ChatOllama(model=MODEL_NAME, 
                             temperature=TEMPERATURE,
                             num_ctx=2048)
            response = llm.invoke(messages)
            return response.content
        
        else:
            logger.warning("No context documents retrieved; generating response without context.")
            return None

    def user_interaction(self, user_query: str = "What is privacy?") -> str:
        """
        End-to-end RAG query:
        1. Retrieve relevant documents
        2. Generate answer from LLM

        Args:
            user_query (str): User input query.

        Returns:
            str: Generated answer.
        """
        new_query = str(input("\n\nPlease ask your question: "))
        if new_query: 
            user_query = new_query
        else:
            logger.info("No input provided. Using default query.")
        context_docs = self.filter_retrieved_context(query=user_query)
        response = self.generate_response(query=user_query, context_docs=context_docs)
        if response:
            return response
        else:
            logger.info(f"\nNo relevant response is found. Please try again")
            return None