FROM python:3.10-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the embedding model into the image at build time, so
# containers never need to hit Hugging Face Hub at runtime (this is what
# was causing 429 rate-limit crashes on cold start).
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# Force offline mode at runtime -- guarantees no network calls to HF Hub,
# even as a fallback, so this failure mode can't recur.
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1

COPY . .

CMD exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}