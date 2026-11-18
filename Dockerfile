FROM python:3.11-slim
WORKDIR /app
COPY requirements.lock ./
RUN pip install --no-cache-dir --require-hashes -r requirements.lock
COPY researchlens ./researchlens
COPY data/papers.jsonl data/reviews.jsonl data/passages.npz data/provenance.json ./data/
COPY data/conferences/papers.jsonl data/conferences/manifest.json ./data/conferences/
COPY artifacts/results.json ./artifacts/results.json
RUN useradd --create-home researchlens && mkdir -p /app/state && chown -R researchlens:researchlens /app/state
USER researchlens
ENV RESEARCHLENS_DB=/app/state/researchlens.db
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')"
CMD ["uvicorn", "researchlens.api:app", "--host", "0.0.0.0", "--port", "8000"]
