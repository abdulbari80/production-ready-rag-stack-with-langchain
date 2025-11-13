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
REL_FACTOR = settings.relative_factor
CTX_WINDOW_SIZE = settings.context_window_size
TOP_K = settings.top_k

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

    # def add_documents(self, documents: List[Document]):
    #     """
    #     Add documents to the vector store.

    #     Args:
    #         documents (List[Document]): List of documents to add.
    #     """
    #     if not documents:
    #         logger.warning("No documents provided to add to the vector store.")
    #         return

    #     if self.vector_store.is_loaded:
    #         # Merge new docs into existing store
    #         logger.info(f"Adding {len(documents)} documents to existing FAISS store...")
    #         try:
    #             self.vector_store.vector_store.add_documents(documents)
    #             self.vector_store.vector_store.save_local(self.vector_store.store_dir)
    #             logger.info("Documents successfully added and FAISS index updated.")
    #         except Exception as e:
    #             logger.exception("Failed to add documents to FAISS store.")
    #             raise
    #     else:
    #         # Create new store
    #         logger.info("FAISS store not initialized. Creating new store...")
    #         self.vector_store.create_store(documents)

    def filter_retrieved_context(self, query: str, top_k: int = TOP_K,
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

        # Extract distances for analysis
        scores = [score for _, score in sim_results]
        best_score = min(scores)
        logger.info(f"Best FAISS distance score: {best_score:.4f}, \
                    cosine ~ {1 - best_score/2:.4f}")

        
        filtered_context_docs = []
        cut_off = 0.0
        if sim_results:
            top_score = min(score for _, score in sim_results)
            if top_score < 1.65:
                cut_off = min(top_score * REL_FACTOR, threshold)
                filtered_context_docs = [doc for doc, score in sim_results if score < cut_off]
            else:
                logger.info("Top document score exceeds threshold; no context will be used.")
        else:
            logger.info("No similar documents found in vector store.")
            
        return filtered_context_docs

    def generate_response(self, query: str, context_docs: List[Document]) -> str:
        """
        Generate a response using the LLM with or without retrieved context.

        Args:
            query (str): User query.
            context_docs (List[Document]): List of documents retrieved as context.

        Returns:
            str: LLM-generated response, with note if no context was used.
        """
        try:
            if context_docs:
                logger.info(f"Generating LLM response using {len(context_docs)} context documents.")
                retrieved_context = "\n\n".join([doc.page_content for doc in context_docs])
                context_note = ""
                system_prompt = (
                    "You're an expert assistant expected to provide responses to the user "
                    "queries on Australian Privacy laws, regulations, frameworks, policies "
                    "and best practices. Please use the context to answer the question."
                )
            else:
                logger.info("No context documents found; generating LLM response without context.")
                retrieved_context = ""
                context_note = "Note: No relevant internal documents found.\n\n"
                system_prompt = (
                    "You're an expert assistant expected to provide responses to the user "
                    "queries on Australian Privacy laws, regulations, frameworks, policies "
                    "and best practices. Answer based on your own knowledge."
                )

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"{retrieved_context}\n\nQuestion: {query}")
            ]

            logger.info("Invoking LLM now...")
            llm = ChatOllama(model=self.llm_model_name,
                            temperature=TEMPERATURE,
                            num_ctx=CTX_WINDOW_SIZE)
            response_obj = llm.invoke(messages)

            if response_obj and response_obj.content:
                logger.info("LLM successfully generated a response.")
                return context_note + response_obj.content
            else:
                logger.warning("LLM did not generate any response.")
                return context_note + "I'm sorry, no response was generated."

        except Exception as e:
            logger.exception("Error while generating response from LLM.")
            return "An error occurred while generating the response."


    def user_interaction(self, user_query: str, top_k: int = TOP_K,
                        threshold: float = THRESHOLD,
                        return_context: bool = True) -> Tuple[str, List[Document]]:
        """
        End-to-end RAG query: retrieve context and generate LLM answer.

        Args:
            user_query (str): User's query string.
            top_k (int): Number of top documents to retrieve.
            threshold (float): Distance threshold for filtering documents (smaller is closer).
            return_context (bool): Whether to return retrieved context documents.

        Returns:
            Tuple[str, List[Document]]: Generated answer and list of retrieved context documents.
        """
        if not user_query or not user_query.strip():
            logger.info("Empty user query received.")
            return "Please provide a valid question.", []

        try:
            logger.info(f"Filtering top {top_k} context documents with threshold {threshold}...")
            context_docs = self.filter_retrieved_context(user_query, top_k=top_k, threshold=threshold)
            logger.info(f"Number of context documents after filtering: {len(context_docs)}")

            response = self.generate_response(user_query, context_docs)

            if return_context:
                return response, context_docs
            else:
                return response, []

        except VectorStoreNotInitializedError:
            logger.error("Vector store not initialized. Falling back to LLM without context.")
            response = self.generate_response(user_query, [])
            return response, []

        except Exception as e:
            logger.exception("Error during user interaction.")
            return "An error occurred while processing your query.", []
