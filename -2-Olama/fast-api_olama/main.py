from fastapi import FastAPI
import uvicorn
from data_models import Message
app = FastAPI()
from ollama import chat


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


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)