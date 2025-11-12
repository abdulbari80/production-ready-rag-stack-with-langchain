from typing import List, Tuple, Optional
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
CTX_WINDOW_SIZE = settings.context_window_size

logger = get_logger(__name__)

class RAGPipeline:
    """
    Retrieval-Augmented Generation (RAG) pipeline.
    Combines a vector store for retrieval with an LLM for response generation.
    """

    def __init__(
        self,
        vector_store: Optional[FaissStore] = None,
        llm_model_name: Optional[str] = None):
        """
        Initialize RAGPipeline.

        Args:
            vector_store (Optional[FaissStore]): Pre-initialized FAISS store instance.
            top_k (int): Number of top documents to retrieve for context.
            llm_model_name (Optional[str]): Optional LLM model name for response generation.
        """
        self.vector_store = vector_store or FaissStore()
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

    def filter_retrieved_context(self, query: str, top_k: int = 3,
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

        sim_results = self.vector_store.similarity_search(query, k=top_k) 
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
            You're an expert assitant and expected to provide responses to the user 
            queries on Australian Privacy laws, regulations, frameworks, policies 
            and best practices. Please use the context to answer the question.
            """
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"{retrieved_context}\n\nQuestion: {query}")
            ]
            llm = ChatOllama(model=MODEL_NAME, 
                             temperature=TEMPERATURE,
                             num_ctx=CTX_WINDOW_SIZE)
            response = llm.invoke(messages)
            return response.content
        
        else:
            logger.warning("No context documents retrieved; generating response without context.")
            return None

    def user_interaction(self, user_query: str, top_k: int = 3,
                         return_context: bool = True) -> Tuple[str, List[Document]]:
        """
        End-to-end RAG query:
        1. Retrieve relevant documents
        2. Generate answer from LLM

        Args:
            user_query (str): User input query.
            return_context (bool): Whether to return retrieved context documents.

        Returns:
            tuple(str: Generated answer, List[Document]: Retrieved context documents).
        """
        
        if user_query:
            context_docs = self.filter_retrieved_context(user_query, 
                                                         top_k=top_k,
                                                         threshold=THRESHOLD)
            response = self.generate_response(user_query, context_docs)
        else:
            logger.info("No input provided. Using default query.")
        
        if response and return_context:
            return (response, context_docs)
        elif response:
            return (response, [])
        else:
            logger.info(f"No relevant response is found. Please try again")
            response = "I'm sorry, no relevant information is found. Please rephrase your question."
            return (response, [])