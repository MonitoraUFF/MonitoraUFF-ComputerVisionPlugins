FROM nvidia/cuda:11.4.3-cudnn8-devel-ubuntu20.04

# Set some variables
ARG CUDA_ARCH_BIN=7.5,8.0,8.6
ARG OPENCV_VERSION=4.5.3

# Install prerequisites
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y software-properties-common && \
    add-apt-repository "deb http://security.ubuntu.com/ubuntu xenial-security main" && \
    add-apt-repository universe

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
        # Generic tools
        build-essential \
        cmake \
        git \
        pkg-config \
        unzip \
        wget \
        yasm \
        zlib1g-dev \
        # Python
        python3-dev \
        python3-pip \
        python3-setuptools \
        # Image I/O libs
        libjasper-dev \
        libjpeg-dev \
        libtiff-dev \
        libpng-dev \
        # Video/Audio libs
        libavcodec-dev \
        libavformat-dev \
        libavresample-dev \
        libavutil-dev \
        libfaac-dev \
        libgstreamer1.0-dev \
        libgstreamer-plugins-base1.0-dev \
        libmp3lame-dev \
        libpostproc-dev \
        libswscale-dev \
        libtheora-dev \
        libv4l-dev \
        libvorbis-dev \
        x264 \
        libx264-dev \
        libxvidcore-dev \
        # Adaptive Multi Rate Narrow Band (AMRNB) and Wide Band (AMRWB) speech codec
        libopencore-amrnb-dev \
        libopencore-amrwb-dev \
        # File transfer and log
        curl \
        libcurl3-dev \
        liblog4cplus-dev \
        # Parallelism library for CPU
        libtbb2 \
        libtbb-dev \
        # Optimization libraries for OpenCV
        gfortran \
        libatlas-base-dev \
        # Other libraries for OpenCV
        doxygen \
        libeigen3-dev \
        libgflags-dev \
        libgphoto2-dev \
        libgoogle-glog-dev \
        libhdf5-dev \
        protobuf-compiler \
        libprotobuf-dev \
        libxine2-dev \
        # Image processing
        libleptonica-dev \
        libopencv-dev \
        && \
    rm -rf /var/lib/apt/lists/*

RUN pip install \
        # Image processing
        opencv-cuda \
        # Machine learning
        scipy \
        torch \
        ultralytics \
        # Optical Character Recognition (OCR)
        easyocr

# Make the plugins ready to use
RUN git clone https://github.com/MonitoraUFF/MonitoraUFF-ComputerVisionPlugins.git /app
VOLUME /app

WORKDIR /app
ENTRYPOINT ["python3", "-m"]
# ENTRYPOINT ["bash"]