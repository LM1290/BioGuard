FROM python:3.10-slim AS builder

# Install core build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create isolated virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install heavy Bio-ML dependencies
# (Includes torch, torch-geometric, rdkit, fastapi, celery, redis)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


FROM nvidia/cuda:12.1.0-base-ubuntu22.04

# Install bare-minimum Python runtime and RDKit geometry rendering libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Alias python3 to python
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# Copy the pre-compiled environment from the builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set up BioGuard working directory
WORKDIR /app
COPY . /app/

ENV PYTHONPATH="/app:${PYTHONPATH}"

# Default command (overridden by docker-compose for workers)
CMD ["uvicorn", "bioguard.api:app", "--host", "0.0.0.0", "--port", "8000"]