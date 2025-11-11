from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    data_dir: str = "data"
    chunk_size : int = 200
    chunk_overlap: int = 50
    faiss_index_dir: str = "vector_store/faiss_index"
    embedding_model_name: str = "all-MiniLM-L6-v2"
    llm_model_name: str = "gemma:2b"
    top_k: int = 3

settings = Settings()

