# Use an official RunPod base image
FROM runpod/pytorch:0.7.0-cu1241-torch240-ubuntu2004

# Environment variables
ENV PYTHONUNBUFFERED=1 

# Modes can be: pod, serverless
ARG MODE_TO_RUN=pod
ENV MODE_TO_RUN=$MODE_TO_RUN

# QWEN Model configuration
ENV MODEL_NAME="Qwen/Qwen3-14B"
ENV USE_QUANTIZATION="true"
ENV DEVICE_MAP="auto"
ENV TRANSFORMERS_CACHE="/app/cache"
ENV HF_HOME="/app/cache"

# Set up the working directory
ARG WORKSPACE_DIR=/app
ENV WORKSPACE_DIR=${WORKSPACE_DIR}
WORKDIR $WORKSPACE_DIR

# Install dependencies in a single RUN command to reduce layers
# Clean up in the same layer to reduce image size
RUN apt-get update --yes --quiet && \
    DEBIAN_FRONTEND=noninteractive apt-get install --yes --quiet --no-install-recommends \
    software-properties-common \
    gpg-agent \
    build-essential \
    apt-utils \
    ca-certificates \
    curl && \
    add-apt-repository --yes ppa:deadsnakes/ppa && \
    apt-get update --yes --quiet && \
    DEBIAN_FRONTEND=noninteractive apt-get install --yes --quiet --no-install-recommends \
    python3-venv \
    python3-pip

# Create and activate a Python virtual environment
RUN python3 -m venv /app/venv
ENV PATH="/app/venv/bin:$PATH"

# Install Python packages
RUN pip install --no-cache-dir \
    asyncio \
    requests \
    runpod

# Install requirements.txt
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt
    
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

# Make sure start.sh is executable
RUN chmod +x start.sh

# Make sure that the start.sh is in the path
RUN ls -la $WORKSPACE_DIR/start.sh

# depot build -t justinrunpod/pod-server-base:1.0 . --push --platform linux/amd64
CMD $WORKSPACE_DIR/start.sh