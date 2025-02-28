from pydantic import BaseModel

class Query(BaseModel):
    session_id: str
    text: str

class Response(BaseModel):
    response: str
