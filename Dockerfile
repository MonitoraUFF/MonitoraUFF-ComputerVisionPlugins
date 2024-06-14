FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu20.04

# Install prerequisites
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
        software-properties-common && \
    add-apt-repository universe && \
    add-apt-repository ppa:deadsnakes/ppa

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
        python3.8 \
        # Image I/O libs
        libjpeg-dev \
        libtiff-dev \
        libpng-dev \
        # Video/Audio libs
        ffmpeg \
        libavcodec-dev \
        libavformat-dev \
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
        # Image processing
        libleptonica-dev \
        && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1 && \
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3.8 get-pip.py

# Install PyTorch and Torchvision
RUN pip3 install \
        torch \
        torchvision \
        torchaudio \
        --index-url https://download.pytorch.org/whl/cu121

# # Install YOLO
# RUN pip3 install \
#         ultralytics

# Install EasyOCR
# RUN DEBIAN_FRONTEND=noninteractive apt-get install -y \
#         libxrender-dev

# RUN git clone https://github.com/JaidedAI/EasyOCR.git /opt/easyocr

# WORKDIR /opt/easyocr
# RUN python3 setup.py build_ext --inplace -j 4 && \
#     pip3 install -e .

# WORKDIR /
# RUN mkdir ~/.EasyOCR && \
#     mkdir ~/.EasyOCR/model && \
#     wget --quiet https://github.com/JaidedAI/EasyOCR/releases/download/v1.3/english_g2.zip && \
#     unzip -qq english_g2.zip -d ~/.EasyOCR/model && \
#     rm english_g2.zip && \
#     wget --quiet https://github.com/JaidedAI/EasyOCR/releases/download/pre-v1.1.6/craft_mlt_25k.zip && \
#     unzip -qq craft_mlt_25k.zip -d ~/.EasyOCR/model && \
#     rm craft_mlt_25k.zip

# # Install OpenCV
# RUN DEBIAN_FRONTEND=noninteractive apt-get install -y \
#         # Optimization libraries for OpenCV
#         gfortran \
#         libatlas-base-dev \
#         # Other libraries for OpenCV
#         doxygen \
#         libeigen3-dev \
#         libgflags-dev \
#         libgphoto2-dev \
#         libgoogle-glog-dev \
#         libhdf5-dev \
#         protobuf-compiler \
#         libprotobuf-dev \
#         libxine2-dev \
#         # OpenCV
#         libopencv-dev \
#         libopencv-contrib-dev

# RUN pip3 install \
#         opencv-python \
#         opencv-contrib-python

# # Cleanup
# RUN apt-get clean -y && \
#     rm -rf /var/lib/apt/lists/*
        
# # Make the plugins ready to use
# RUN git clone https://github.com/MonitoraUFF/MonitoraUFF-ComputerVisionPlugins.git /app
# VOLUME /app

# WORKDIR /app
# ENTRYPOINT ["python3", "-m"]
ENTRYPOINT ["bash"]