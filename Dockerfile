# Dockerfile — HuggingFace Spaces Ready
# =======================================
# HF Spaces requirements:
#   - Port MUST be 7860
#   - Non-root user recommended (HF runs as uid=1000)
#   - Dockerfile must be in project ROOT

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install dependencies first (Docker layer cache — only reinstalls if requirements.txt changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy entire project
COPY . .

# HF Spaces runs as non-root user uid=1000 — create and match it
RUN useradd -m -u 1000 appuser \
    && chown -R appuser:appuser /app

USER appuser

# Expose port 7860 — MANDATORY for HF Spaces
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')" || exit 1

# Start server
# server.app:app means: server/ folder → app.py file → app object
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]