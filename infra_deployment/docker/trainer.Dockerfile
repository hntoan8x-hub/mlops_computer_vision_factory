# Use a base image with CUDA support
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Set up the environment
ENV LANG C.UTF-8
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    # Add other system dependencies if needed
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set up Python environment
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1
RUN python -m pip install --upgrade pip

# Copy project code and install Python dependencies
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project code
COPY . .

# Set entrypoint for training jobs
ENTRYPOINT ["python", "main_training_script.py"]