# 1. Use a specific, stable version
FROM python:3.11-slim

# 2. Set environment variables to prevent Python from buffering logs
# (Critical for seeing errors in Docker logs!)
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0

# 3. System dependencies for RDKit
RUN apt-get update && apt-get install -y \
    build-essential \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 4. Layer caching: Install requirements first
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy project (Ensure you have a .dockerignore to skip the junk!)
COPY . .

# 6. Ensure the directories exist
RUN mkdir -p data artifacts

EXPOSE 8501

# 7. The "Safe" Start: Validate that we have a model before launching
CMD ["sh", "-c", "if [ ! -f artifacts/model.pt ]; then echo 'ERROR: Model weights not found in /artifacts'; exit 1; fi; streamlit run streamlit/app.py"]
