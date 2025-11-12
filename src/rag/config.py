from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    data_dir: str = "data/privacy"
    chunk_size : int = 500
    chunk_overlap: int = 100
    faiss_index_dir: str = "vector_store_au/faiss_index"
    embedding_model_name: str = "all-MiniLM-L6-v2"
    llm_model_name: str = "llama3.2:3b"
    top_k: int = 3
    llm_temperature: float = 0.2
    threshold: float = 0.5

settings = Settings()

