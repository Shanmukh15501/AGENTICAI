from qdrant_client import QdrantClient
from qdrant_client.http import models
import uuid
from qdrant_client.http.models import Distance, VectorParams






class VectorDB:
    def __init__(self):
        self.client = QdrantClient(host="localhost", port=6333)
        self.collection_name = "my_collection"
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=3072, distance=Distance.COSINE),
            )


        
    def insert(self, embeddings, chunks):
        """Insert embeddings + metadata into Qdrant"""
        points = [
            models.PointStruct(
                id=str(uuid.uuid4()),  # unique ID
                vector=embeddings[i],
                payload={"text": chunks[i]}  # store chunk text as metadata
            )
            for i in range(len(embeddings))
        ]

        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
                
        print(f"✅ Inserted {len(points)} vectors into {self.collection_name}")
