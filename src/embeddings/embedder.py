from langchain_community.embeddings import SentenceTransformerEmbeddings
from typing import List

class Embeddings:
    """This embeds documents into high dim vectors"""
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model_name = model_name
        self.model = SentenceTransformerEmbeddings(model_name = self.model_name)

    def embed_documents(self, texts:List[str]) -> List[List[float]]:
        """Embeds source texts

        parameter:
            list[str]: list of text documents

        return:
            list[list[float]]: list of embeddd high dimensional vectors

        """
        return self.model.embed_documents(texts=texts)
    
    def embed_query(self, query: str) -> List[float]:
        """Embeds query text"""
        return self.model.embed_query(text=query)