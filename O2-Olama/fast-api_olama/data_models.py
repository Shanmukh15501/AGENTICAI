from pydantic import BaseModel


class Message(BaseModel):
    user_input: str