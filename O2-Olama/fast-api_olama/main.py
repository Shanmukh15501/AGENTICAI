from fastapi import FastAPI,UploadFile
import uvicorn
from data_models import Message
app = FastAPI()
from ollama import chat
from rag_pipeline import RecursiveChunker,OpenAIEmbedding,DocumentLoader,VectorDB,RAGPipeline
import shutil
import os
from pathlib import Path
from worker import send_message_queue
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
from redis import Redis
from rq import Queue

q = Queue(connection=Redis(host="localhost",port="6379"))


chunking_instance =  RecursiveChunker()
embedding_instance =  OpenAIEmbedding()
load_instance = DocumentLoader()
vector_db_instance = VectorDB()
pipeline_instance = RAGPipeline(load_instance, chunking_instance, embedding_instance, vector_db_instance)


#http://127.0.0.1:8000/redoc--docs swagger 

@app.get("/")
def read_root():
    return {"message": "Hello World"}

@app.post("/chat/")
async def send_message(msg: Message):
    """
    This Function helps to interact with the ollama models
    """
    response  = chat(model='gemma:2b', messages=[
        {
            'role': 'user',
            'content': msg.user_input
        },
    ])
    return {
            "status":"success",
            "response":response.get('message',{}).get("content","no response")    
            }



@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile):
    # Sanitize and build safe file path
    safe_filename = os.path.basename(file.filename)
    file_path = UPLOAD_DIR / safe_filename

    # Save file in chunks (efficient, handles large files safely)
    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # ✅ Now pass file_path to your RAG pipeline here if needed
    pipeline_instance.ingest(str(file_path))

    return {"file_path": str(file_path)}


@app.post("/chat-service")
async def send_message(msg: Message):
    """
    This Function interacts with the RAG pipeline and OpenAI ChatCompletion
    """
    # Step 1: Retrieve relevant context from your pipeline
    retrieved_docs = pipeline_instance.retrieve(msg.user_input)
    
    context = ""
    for doc in retrieved_docs:
        context += doc.payload.get('text', {}).get('page_content','') + "\n\n"

    # Step 2: Construct prompt
    prompt = f"""You are a helpful assistant. Use the following context to answer the user's question.

                Context:
                {context}

                Question:
                {msg.user_input}
        """

    # Step 3: Call OpenAI ChatCompletion
    from openai import OpenAI
    
    client = OpenAI()
    chat_response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                                {"role": "system", "content": "You are a helpful assistant."},
                                {"role": "user", "content": prompt}
                                ]
                            )



    # Step 4: Return the response
    return {
        "status": "success",
        "response": chat_response.choices[0].message.content
    }


@app.post("/async/chat-service")
async def send_async_message(msg: Message):
    """
    This Function interacts runs the interaction of rag_pipeline in a redis server
    """
    print("come")
    job = q.enqueue(send_message_queue, msg.user_input)
    

    return {"status":"queued","job":job.id}


@app.get("/job/{job_id}")
def get_job_result(job_id: str):
    job = q.fetch_job(job_id)
    print("job",job)
    if job == None:
        return {"status": "Still Processing", "message": job}
    return {"status": "finished", "message": job.return_value()}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)