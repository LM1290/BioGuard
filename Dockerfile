# Use Python 3.11 slim for a smaller footprint
FROM python:3.11-slim

# Install system dependencies required by RDKit
RUN apt-get update && apt-get install -y \
    build-essential \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy requirements first to leverage Docker layer caching
COPY requirements_training.txt .
RUN pip install --no-cache-dir -r requirements_training.txt

# Copy the entire project
# This includes the 'artifacts/' folder which contains your trained model
COPY . .

# Ensure data and artifacts directories exist
RUN mkdir -p data artifacts

# Expose the Streamlit default port
EXPOSE 8501

# Run the Streamlit app on container start
CMD ["streamlit", "run", "streamlit/app.py", "--server.port=8501", "--server.address=0.0.0.0"]