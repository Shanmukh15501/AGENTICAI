from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from typing import List
from abc import ABC,abstractmethod
from fastapi import UploadFile


class BaseLoader(ABC):
    """
    Base Class for all Loaders
    """
    @abstractmethod
    def load(self, file_path: str) -> List[Document]:
        """
        Loads the documents from the path and the given type
        
        Args :- 
            file_path : str (The path of the file)
        
        Return :-
            The list of loaded documents : List[Document]
        """
        pass

class DocumentLoader(BaseLoader):
    """
    Responsible for loading documents from various file types.
    """
    def load(self, file: str) -> List[Document]:
        if file.endswith(".pdf"):
            loader = PyPDFLoader(file)
            return loader.load()
        raise ValueError(f"Unsupported file type: {file}")