from abc import ABC, abstractmethod
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

class BaseChunker(ABC):
    """
    Abstract base class for all chunking strategies.
    """
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    @abstractmethod
    def chunk(self, documents: List[Document]) -> List[Document]:
        pass

class RecursiveChunker(BaseChunker):
    """
    Uses RecursiveCharacterTextSplitter for chunking.
    """
    def chunk(self, documents: List[Document]) -> List[Document]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        return splitter.split_documents(documents)

