# ── Stage 1: Builder ──────────────────────────────────────────────────────────
# Install dependencies in a separate layer so they're cached independently
# of application code changes.
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build tools needed for some Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --prefix=/install --no-cache-dir -r requirements.txt


# ── Stage 2: Runtime ──────────────────────────────────────────────────────────
# Lean final image: only runtime deps, no build tools.
# Result: ~800MB vs ~3GB for a naive single-stage build.
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY src/         ./src/
COPY app.py       .
COPY api.py       .

# Create non-root user (security best practice)
RUN useradd --create-home appuser && \
    mkdir -p data/ vectorstore/ && \
    chown -R appuser:appuser /app

USER appuser

# Expose both Streamlit (8501) and FastAPI (8000) ports
EXPOSE 8501 8000

# Default: start FastAPI server
# Override with: docker run ... streamlit run app.py --server.port 8501
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
