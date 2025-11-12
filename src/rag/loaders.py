from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List
from pathlib import Path
from .config import settings
from src.rag.logger import get_logger

logger = get_logger(__name__)

CHUNK_SIZE = settings.chunk_size
CHUNK_OVERLAP = settings.chunk_overlap

class TxtLoader:
    """This loads TXT file and splits into overlapped chunks"""

    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap:int = CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = self.chunk_size,
            chunk_overlap = self.chunk_overlap
        )

    def load_txt(self, dir_path: str) -> List[Document]:
        """Loads a set of TXT files from a given directory and splits into chunks.
        parameter:
            dir_path (str): directory path where TXT files are stored
        return:
            List[Document]: list of splitted document chunks
        """
        documents = []
        txt_files = Path(dir_path).glob('*.txt')
        i = 0
        for file in txt_files:
            i += 1
            loader = TextLoader(file_path=str(file))
            document = loader.load()
            documents.extend(document)
            logger.info(f"Section {i} with {len(document)} pages loaded .....")
        chunks = self.text_splitter.split_documents(documents)

        return chunks
