# server/Dockerfile
# =================
# Builds the container that runs the Email Triage RL environment server.
#
# Build:  docker build -t email-triage-env .
# Run:    docker run -p 8000:8000 email-triage-env
# Test:   curl http://localhost:8000/health

FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Copy requirements first (Docker cache optimization — only re-installs when requirements.txt changes)
COPY ../requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy server files
COPY . /app/server/

# Copy models.py and data/ from root (server needs them)
COPY ../models.py /app/models.py
COPY ../data/ /app/data/

# Expose FastAPI port
EXPOSE 8000

# Start the server
# --host 0.0.0.0 makes it accessible from outside the container
# Remove --reload in production (only for dev)
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
