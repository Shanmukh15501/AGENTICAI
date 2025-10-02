from langchain_qdrant import QdrantVectorStore

class RAGPipeline:
    def __init__(self, loader, chunker, embedder, vectorstore):
        self.loader = loader
        self.chunker = chunker
        self.embedder = embedder
        self.vdb = vectorstore

    def ingest(self, file):
        docs = self.loader.load(file)
        chunks = self.chunker.chunk(docs)        
        embeddings = self.embedder.embed(chunks)        
        self.vdb.insert(embeddings,chunks)
        

    def retrieve(self, query):
        query_embedding = self.embedder.embed(query)
        hits = self.vdb.client.search(
            collection_name="my_collection",
            query_vector=query_embedding,
            limit=3  # Return 5 closest points
            )
        return hits
