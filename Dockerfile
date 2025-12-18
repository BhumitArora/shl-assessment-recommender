# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY step5_api/ ./step5_api/
COPY data/processed_assessments.csv ./data/
COPY data/assessment_embeddings_google.npy ./data/
COPY frontend/ ./frontend/

# Create empty .env file if needed
RUN touch .env

# Expose port
EXPOSE 8000

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Run the application
CMD ["uvicorn", "step5_api.main:app", "--host", "0.0.0.0", "--port", "8000"]

