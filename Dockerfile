FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy dependency file first (Docker layer caching — only reinstalls if requirements.txt changes)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy entire project (all files from root)
COPY . .

# Expose port 7860 (HF Spaces standard)
EXPOSE 7860

# Health check (optional but recommended)
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
  CMD python -c "import requests; requests.get('http://localhost:7860/health', timeout=5)" || exit 1

# Start FastAPI server
# Note: HF Spaces sets environment variables automatically
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]