from data_models import Message
from rag_pipeline import RecursiveChunker,OpenAIEmbedding,DocumentLoader,VectorDB,RAGPipeline
from pathlib import Path
from redis import Redis
from rq import Queue
from openai import OpenAI

q = Queue(connection=Redis(host="localhost",port="6379"))

chunking_instance =  RecursiveChunker()
embedding_instance =  OpenAIEmbedding()
load_instance = DocumentLoader()
vector_db_instance = VectorDB()

pipeline_instance = RAGPipeline(load_instance, chunking_instance, embedding_instance, vector_db_instance)

def send_message_queue(msg):
    
    print("coming here",msg)
    retrieved_docs = pipeline_instance.retrieve(msg)
    context = "\n\n".join([
        doc.payload.get('text', {}).get('page_content', '') for doc in retrieved_docs
    ])

    prompt = f"""You are a helpful assistant. Use the following context to answer the user's question.

    Context:
    {context}

    Question:
    {msg}
    """

    client = OpenAI()
    chat_response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )

    return chat_response.choices[0].message.content