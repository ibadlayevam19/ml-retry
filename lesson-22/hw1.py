from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="FastAPI Fundamentals Demo")

# Pydantic model for POST /echo
class EchoRequest(BaseModel):
    message: str
    count: int | None = None


@app.get("/")
def read_root():
    return {
        "description": "This is a simple FastAPI application demonstrating basic endpoints."
    }


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/echo")
def echo_data(data: EchoRequest):
    return data
