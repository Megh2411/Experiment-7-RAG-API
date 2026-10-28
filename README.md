# Experiment 7 — Production-oriented Customer Support RAG API

## What is included
- `app.py`: FastAPI `/health` and `/ask` endpoints, input validation, controlled errors, logging, and latency in milliseconds.
- `support_rag.py`: isolated support-answer logic and sample knowledge base.
- `tests/test_api.py`: automated endpoint tests.
- `.env.example`: safe configuration template; no secrets are included.

**Integration note:** The uploaded Experiment 5 notebook defines the wellness, clinic FAQ, and medication-safety content inline, and uses TXT/PDF/DOCX loaders, 500-character chunks with 80-character overlap, `all-MiniLM-L6-v2` embeddings, FAISS top-3 retrieval, and `Qwen/Qwen2.5-0.5B-Instruct`. This package follows that design in `support_rag.py` and includes recreated versions of the three knowledge-base files under `data/`, based on the content written in the notebook. The original separately uploaded TXT/PDF/DOCX files were not supplied, so the files in `data/` are reconstructed equivalents, not byte-for-byte originals.

The default `RAG_BACKEND=mock` lets you test API structure without downloading model weights. Set `RAG_BACKEND=local` to run the Experiment 5-style local RAG pipeline. The first local request downloads/loads the embedding and language models, so it can take time and requires sufficient memory and disk space.

## Run locally (Python 3.10+)
```bash
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows:
# .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app:app --reload
```
Open <http://127.0.0.1:8000/docs>.

## Test examples
- `GET /health`
- `POST /ask` body: `{"question":"What does the wellness note say about hydration?"}`
- `POST /ask` body: `{"question":"What is the capital of Mars?"}` (should return an uncertainty fallback)
- Empty/whitespace and >1000-character questions should be rejected with HTTP 422.

## Optional local LLM backend (Ollama)
Install Ollama, then run:
```bash
ollama pull qwen2.5:0.5b
ollama serve
```
In another terminal:
```bash
export RAG_BACKEND=ollama
export OLLAMA_MODEL=qwen2.5:0.5b
uvicorn app:app --reload
```
On Windows PowerShell use `$env:RAG_BACKEND="ollama"` and `$env:OLLAMA_MODEL="qwen2.5:0.5b"`.

The Ollama route uses the same sample knowledge base as context. It is not a substitute for integrating the actual Experiment 6 retriever.

## Automated tests
```bash
pytest -q
```

## Security notes
- Never commit `.env` or API keys.
- This is a lab/demo, not a fully production-ready public service.
- `/health` confirms the API process is responding, not that the model is available.


## Use the Experiment 5 local RAG pipeline
Install the full requirements, then set the backend before starting FastAPI.

macOS/Linux:
```bash
export RAG_BACKEND=local
uvicorn app:app --reload
```

Windows PowerShell:
```powershell
$env:RAG_BACKEND="local"
uvicorn app:app --reload
```

The first request may download the embedding model and Qwen model from Hugging Face. For a quick endpoint demonstration, leave `RAG_BACKEND=mock`.
