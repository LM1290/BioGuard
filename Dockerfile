# Use Python 3.11 as the base image (compatible per requirements.txt notes)
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
# libxrender1 and libxext6 are often required for RDKit
RUN apt-get update && apt-get install -y \
    build-essential \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Copy the production requirements file
COPY requirements.txt .

# Install Dependencies with PyG Support
# 1. Install PyTorch CPU version first to keep image size down
# 2. Install requirements using the specific PyG wheel index for Torch 2.1.0 + CPU
RUN pip install --no-cache-dir torch==2.1.0 --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt \
    -f https://data.pyg.org/whl/torch-2.1.0+cpu.html

# Copy the application code
COPY . .

# Expose port 8000
EXPOSE 8000

# Command to run the application
CMD ["python", "-m", "bioguard.main", "serve"]