from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List
from pathlib import Path
from .config import settings

CHUNK_SIZE = settings.chunk_size
CHUNK_OVERLAP = settings.chunk_overlap

class PDFLoader:
    """This loads PDF file and splits into overlapped chunks"""

    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap:int = CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = self.chunk_size,
            chunk_overlap = self.chunk_overlap
        )

    def load_pdf(self, dir_path: str) -> List[Document]:
        """Loads a set of PDF files from a given directory and splits into chunks.
        parameter:
            dir_path (str): directory path where PDF files are stored
        return:
            List[Document]: list of splitted document chunks
        """
        documents = []
        pdf_files = Path(dir_path).glob('*.pdf')
        for pdf_file in pdf_files:
            loader = PyPDFLoader(file_path=str(pdf_file))
            document = loader.load()
            documents.extend(document)
        chunks = self.text_splitter.split_documents(documents)

        return chunks
