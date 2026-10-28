"""
Experiment 7: Preparing an LLM Application for Production-Ready Deployment
FastAPI wrapper with validation, safe errors, logging, and latency measurement.

Default backend is a deterministic mock so the API can be tested without a key,
network access, or a local model. Set RAG_BACKEND=ollama to use Ollama.
"""
import logging
import os
import time
from typing import Literal

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from support_rag import answer_question

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("experiment7")

app = FastAPI(
    title="Customer Support RAG API",
    description="Experiment 7 API with validation, controlled errors, and latency.",
    version="1.0.0",
)

class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Question must not be empty or whitespace.")
        return value.strip()

class AskResponse(BaseModel):
    answer: str
    latency_ms: float
    backend: str

@app.get("/health")
def health():
    """Reports whether the API process is running; does not guarantee model health."""
    return {"status": "ok"}

@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    started = time.perf_counter()
    backend = os.getenv("RAG_BACKEND", "mock").lower()
    try:
        answer = answer_question(request.question)
        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        logger.info("ask_success backend=%s latency_ms=%.2f", backend, latency_ms)
        return AskResponse(answer=answer, latency_ms=latency_ms, backend=backend)
    except (httpx.HTTPError, TimeoutError, ConnectionError) as exc:
        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        logger.warning("ask_backend_failure backend=%s latency_ms=%.2f error_type=%s",
                       backend, latency_ms, type(exc).__name__)
        raise HTTPException(status_code=502, detail="The model backend is unavailable. Please try again later.")
    except Exception as exc:
        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        logger.exception("ask_internal_failure backend=%s latency_ms=%.2f error_type=%s",
                         backend, latency_ms, type(exc).__name__)
        raise HTTPException(status_code=500, detail="Unable to process the question right now.")

@app.get("/")
def root():
    return {"message": "Customer Support RAG API. Open /docs to test."}
