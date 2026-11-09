from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from fastapi.responses import FileResponse

from model_service import ModelServiceError, TwoLoraModelService

PROJECT_DIR = __file__
MODEL_SERVICE = TwoLoraModelService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    MODEL_SERVICE.load()
    app.state.model_service = MODEL_SERVICE
    yield


app = FastAPI(title="Turkish Two-Task Text Assistant", lifespan=lifespan)


@app.get("/", include_in_schema=False)
def home():
    from pathlib import Path

    return FileResponse(Path(PROJECT_DIR).resolve().parent / "static" / "index.html")


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=20_000)
    history: list[ChatMessage] = Field(default_factory=list)
    show_details: bool = False


@app.get("/health")
def health_check():
    return MODEL_SERVICE.health()


@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        result = MODEL_SERVICE.generate(request.message)
    except ModelServiceError as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        ) from exc

    response = {"reply": result["reply"]}
    if request.show_details:
        response["details"] = result["details"]
    return response
