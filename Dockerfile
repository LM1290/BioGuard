# 1. FORCE linux/amd64
FROM --platform=linux/amd64 python:3.11-slim

WORKDIR /app

# 2. System Dependencies
# libxrender1 is required for RDKit
RUN apt-get update && apt-get install -y \
    libxrender1 \
    libxext6 \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# 3. Build Argument (Defaults to CPU for Mac stability)
ARG DEVICE=cpu

# 4. Smart Install Logic
# - LOCAL (CPU): Small files (~200MB), works perfectly on Mac via emulation.
# - PROD (CUDA): Huge files (~3GB), usage on Mac requires High-Mem Docker settings.
RUN pip install --upgrade pip && \
    if [ "$DEVICE" = "cuda" ]; then \
      echo "--- BUILDING FOR PRODUCTION (CUDA 12.1) ---"; \
      pip install --no-cache-dir torch==2.2.0 --index-url https://download.pytorch.org/whl/cu121 && \
      pip install --no-cache-dir torch_geometric pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
      -f https://data.pyg.org/whl/torch-2.2.0+cu121.html; \
    else \
      echo "--- BUILDING FOR LOCAL DEV (CPU) ---"; \
      pip install --no-cache-dir torch==2.2.0 --index-url https://download.pytorch.org/whl/cpu && \
      pip install --no-cache-dir torch_geometric pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
      -f https://data.pyg.org/whl/torch-2.2.0+cpu.html; \
    fi

# 5. Remaining Deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy Code
COPY . .

# 7. Runtime
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "-m", "bioguard.main", "serve"]