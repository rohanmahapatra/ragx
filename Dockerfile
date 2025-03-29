FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Install Miniconda, Vim, and OpenJDK (Java 17)
RUN apt-get update && apt-get install -y \
    wget \
    vim \
    openjdk-17-jdk \
    && wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda \
    && rm Miniconda3-latest-Linux-x86_64.sh \
    && /opt/conda/bin/conda clean -ya \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set environment variables for Miniconda and Java
ENV PATH=/opt/conda/bin:$PATH
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64

# Set working directory
WORKDIR /app

# Install system-level dependencies (required for nmslib, faiss, etc.)
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libstdc++6 \
    libboost-all-dev \
    software-properties-common \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN apt update && apt install -y unzip

# Copy environment and code into container
COPY environment.yml .
COPY . .

# Create the Conda environment
RUN conda env create -f environment.yml

# Conda shell init + environment path
RUN conda init bash
ENV PATH=/opt/conda/envs/artifact/bin:$PATH
RUN echo "conda activate artifact" >> ~/.bashrc

# Set NCCL and FAISS shared library fixes
ENV NCCL_P2P_DISABLE=1
ENV NCCL_IB_DISABLE=1
ENV LD_PRELOAD=/opt/conda/envs/artifact/lib/libstdc++.so.6

# Default entrypoint
CMD ["bash"]
