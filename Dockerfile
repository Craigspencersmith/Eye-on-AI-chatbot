FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies needed by chromadb
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create data directories
RUN mkdir -p chroma_data

# Expose the server port
EXPOSE 8000

# Default command: run the server
CMD ["python", "server.py"]
