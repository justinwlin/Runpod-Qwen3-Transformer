# Use an official RunPod base image with newer Python
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# Environment variables
ENV PYTHONUNBUFFERED=1 

# Modes can be: pod, serverless
ARG MODE_TO_RUN=pod
ENV MODE_TO_RUN=$MODE_TO_RUN

# QWEN Model configuration - can be overridden at build time
ARG MODEL_NAME="Qwen/Qwen3-0.6B"
ENV MODEL_NAME=$MODEL_NAME
ENV USE_QUANTIZATION="true"
ENV DEVICE_MAP="auto"
ENV TRANSFORMERS_CACHE="/app/cache"
ENV HF_HOME="/app/cache"

# Set up the working directory
ARG WORKSPACE_DIR=/app
ENV WORKSPACE_DIR=${WORKSPACE_DIR}
WORKDIR $WORKSPACE_DIR

# Install dependencies in a single RUN command to reduce layers
RUN apt-get update --yes --quiet && \
    DEBIAN_FRONTEND=noninteractive apt-get install --yes --quiet --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    python3-venv \
    python3-pip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Create and activate a Python virtual environment
RUN python3 -m venv /app/venv
ENV PATH="/app/venv/bin:$PATH"

# Install Python packages
RUN pip install --no-cache-dir \
    asyncio \
    requests \
    runpod

# Upgrade pip and install build tools
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Check if nvcc is available (needed for flash-attn compilation)
RUN echo "=== Checking for nvcc (CUDA compiler) ===" && \
    if ! which nvcc; then \
        echo "ERROR: nvcc not found - required for flash-attn compilation"; \
        echo "Please use a 'devel' image that includes CUDA toolkit"; \
        exit 1; \
    fi && \
    nvcc --version

# Install ninja for faster compilation
RUN pip install --no-cache-dir ninja && \
    ninja --version

# Install requirements.txt (excluding flash-attn first)
COPY requirements.txt ./requirements.txt
RUN grep -v "flash-attn" requirements.txt > requirements_no_flash.txt && \
    pip install --no-cache-dir -r requirements_no_flash.txt

# Install flash-attn separately with proper flags
# Using MAX_JOBS to limit memory usage during compilation
RUN MAX_JOBS=4 pip install --no-cache-dir flash-attn --no-build-isolation || \
    echo "Warning: flash-attn installation failed, continuing without it"
    
# Remove Runpod's copy of start.sh and replace it with our own
RUN rm ../start.sh

# Create cache directory for models
RUN mkdir -p /app/cache

# COPY EVERYTHING INTO THE CONTAINER
COPY handler.py $WORKSPACE_DIR/handler.py
COPY start.sh $WORKSPACE_DIR/start.sh
COPY download_model.py $WORKSPACE_DIR/download_model.py

# Pre-download the model during build (prebaking)
RUN python download_model.py

# Verify the correct model was downloaded
RUN echo "=== MODEL VERIFICATION ===" && \
    echo "Expected model: $MODEL_NAME" && \
    echo "HF_HOME: $HF_HOME" && \
    echo "Cache directory contents:" && \
    ls -la $HF_HOME/ && \
    EXPECTED_DIR="models--$(echo $MODEL_NAME | sed 's|/|--|g')" && \
    echo "Expected directory: $EXPECTED_DIR" && \
    echo "Full expected path: $HF_HOME/$EXPECTED_DIR" && \
    if [ -d "$HF_HOME/$EXPECTED_DIR" ]; then \
        echo "✅ Correct model found!"; \
    else \
        echo "❌ Expected model directory not found!"; \
        echo "DEBUG: Trying to find any models..."; \
        find $HF_HOME -name "*models*" -type d || echo "No model directories found"; \
        echo "❌ Build failed - model verification failed"; \
        exit 1; \
    fi && \
    echo "==========================="

# Make sure start.sh is executable
RUN chmod +x start.sh

# Make sure that the start.sh is in the path
RUN ls -la $WORKSPACE_DIR/start.sh

# depot build -t justinrunpod/pod-server-base:1.0 . --push --platform linux/amd64
CMD $WORKSPACE_DIR/start.sh