# 1. Base Image (Architecture Neutral)
FROM python:3.11-slim

WORKDIR /app

# 2. System Dependencies
# We still need these for RDKit (libxrender) and PyTDC (build-essential/git)
RUN apt-get update && apt-get install -y \
    libxrender1 \
    libxext6 \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# 3. SIMPLIFIED INSTALL STRATEGY (User Defined)
# Install pure CPU PyTorch
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install PyG Core ONLY (Uses native PyTorch fallbacks instead of compiled extensions)
RUN pip install torch-geometric
COPY requirements_training.txt .
RUN pip install --no-cache-dir -r requirements_training.txt

# 5. Copy Code
COPY . .

# 6. Runtime
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# 7. Hybrid Entrypoint

ENTRYPOINT ["python", "-m", "bioguard.main"]
CMD ["serve"]