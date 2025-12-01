# Use Python 3.11 (better compatibility with older SciPy)
FROM python:3.11-slim

# Install system dependencies for numpy/scipy
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    gfortran \
    libopenblas-dev \
    liblapack-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Python packages (use compatible versions)
RUN pip install --no-cache-dir \
    numpy==1.24.3 \
    scipy==1.10.1 \
    pandas

# Create directories
RUN mkdir -p /codeBase
RUN mkdir -p /opt

# Copy PyBrain from build context
COPY pybrain /opt/pybrain

# Copy activation script
COPY UE_07_App5.py /codeBase/UE_07_App5.py

# Set working directory
WORKDIR /codeBase

# Set Python path to include PyBrain
ENV PYTHONPATH=/opt/pybrain

# Default command
CMD ["python3", "/codeBase/UE_07_App5.py"]