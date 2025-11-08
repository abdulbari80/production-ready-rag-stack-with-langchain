# src/rag/exceptions.py

class RAGBaseException(Exception):
    """Base class for all custom RAG exceptions."""
    pass


class VectorStoreNotInitializedError(RAGBaseException):
    """Raised when trying to query a vector store that hasn't been loaded."""
    def __init__(self, message="Vector store not initialized or loaded."):
        super().__init__(message)


class DocumentLoadError(RAGBaseException):
    """Raised when document loading/parsing fails."""
    def __init__(self, message="Failed to load or parse documents."):
        super().__init__(message)


class EmbeddingModelError(RAGBaseException):
    """Raised when embedding generation fails."""
    def __init__(self, message="Error generating embeddings from model."):
        super().__init__(message)


class RetrievalError(RAGBaseException):
    """Raised when retrieval or similarity search fails."""
    def __init__(self, message="Error during document retrieval."):
        super().__init__(message)
