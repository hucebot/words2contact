FROM --platform=x86_64 nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04
ARG CUDNN_ARCH="linux-x86_64"

# Set the default shell to bash
SHELL ["/bin/bash", "-c"]
ENV SHELL /bin/bash

# Set the timezone
ENV TZ="Europe/Paris"
ARG DEBIAN_FRONTEND=noninteractive
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

WORKDIR /deps

# Install required packages
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y \
    python3-dev python-is-python3 python3-pip git ninja-build cmake build-essential \
    libopenblas-dev clang unzip wget software-properties-common bash-completion \
    libx11-dev python3-tk neovim xorg-dev autoconf libtool ffmpeg portaudio19-dev \
    python3-opencv python3-pyaudio terminator libxext-dev x11proto-gl-dev curl \
    zlib1g-dev libusb-1.0-0-dev freeglut3-dev liblapacke-dev libatlas-base-dev \
    make lsb-release tree sudo gnupg-agent libssl-dev apt-transport-https \
    gnupg2 usbutils mesa-utils mesa-va-drivers vainfo locales iputils-ping \
    libv4l-dev v4l-utils libnuma-dev libnuma1 libgles2-mesa libeigen3-dev \
    ca-certificates && \
    rm -rf /var/lib/apt/lists/*


# Set CUDA environment variables
ENV CUDA_HOME=/usr/local/cuda-12.1
ENV PATH=$PATH:/usr/local/cuda-12.1/bin/
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6+PTX"
ENV CUDACXX=/usr/local/cuda-12.1/bin/nvcc

# Install Python dependencies
RUN pip install --no-cache-dir \
    numpy==1.23 pandas matplotlib scipy torch torchvision \
    git+https://github.com/huggingface/transformers.git@main \
    git+https://github.com/huggingface/peft.git \
    datasets evaluate==0.4.0 rouge_score==0.1.2 sentencepiece accelerate \
    bitsandbytes peft trl openai ftfy


# Install LLAMA CPP Python with CUDA support
RUN pip install llama-cpp-python \
  --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121

# Download and setup weights for GroundingDINO
RUN mkdir -p /deps/weights && \
    cd /deps/weights && \
    wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth

# Clone and install GroundingDINO
RUN git clone https://github.com/IDEA-Research/GroundingDINO.git /deps/GroundingDINO && \
    cd /deps/GroundingDINO && \
    python -m pip install .

# Fix OpenCV package issue
RUN pip uninstall -y opencv-python-headless && \
    pip install --no-cache-dir --upgrade opencv-python

# Set working directory
WORKDIR /home/words2contact/

# Set default entrypoint
ENTRYPOINT ["/bin/bash"]
