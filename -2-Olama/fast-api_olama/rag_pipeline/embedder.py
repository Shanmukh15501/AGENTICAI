from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from typing import List, Union
from abc import ABC, abstractmethod
from dotenv import load_dotenv
import os
load_dotenv()


os.environ['OPENAI_API_KEY']=os.getenv("OPENAI_API_KEY", "")


class BaseEmbedding(ABC):
    """
    Base Class for all the embeddings
    """
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        
    @abstractmethod
    def embed(self, documents: List[Document]) -> List:
        pass

class OpenAIEmbedding(BaseEmbedding):
    """
    Useful to embed the open ai embedding models
    """
    def __init__(self, model_name: str = "text-embedding-3-large"):
        self.model_name  = model_name
    
    def embed(self, data: Union[List[Document]|str]) -> List:
        
        embeddings = OpenAIEmbeddings(model=self.model_name)
        
        if isinstance(data,list):
            return embeddings.embed_documents([doc.page_content for doc in data])
        else:
            return embeddings.embed_query(data)
    
        