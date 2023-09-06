FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

ENV DEBIAN_FRONTEND=nonintercative

ENV CMAKE_VERSION=3.21.0

RUN echo "Installing apt packages..." \
	&& apt -y update --no-install-recommends \
	&& apt -y install --no-install-recommends \
	git \
	wget \
	ffmpeg \
	tk-dev \
	libxi-dev \
	libc6-dev \
	libbz2-dev \
	libffi-dev \
	libomp-dev \
	libssl-dev \
	zlib1g-dev \
	libcgal-dev \
	libgdbm-dev \
	libglew-dev \
	python3-dev \
	python3-pip \
	qtbase5-dev \
	checkinstall \
	libglfw3-dev \
	libeigen3-dev \
	libgflags-dev \
	libxrandr-dev \
	libopenexr-dev \
	libsqlite3-dev \
	libxcursor-dev \
	build-essential \
	libcgal-qt5-dev \
	libxinerama-dev \
	libboost-all-dev \
	libfreeimage-dev \
	libncursesw5-dev \
	libatlas-base-dev \
	libqt5opengl5-dev \
	libgoogle-glog-dev \
	libsuitesparse-dev \
	python3-setuptools \
	libreadline-gplv2-dev \
	&& apt autoremove -y \
	&& apt clean -y \
	&& export DEBIAN_FRONTEND=dialog

ENV PATH="/usr/bin/cmake/bin:${PATH}"
ENV CUDA_HOME=/usr/local/cuda
ENV LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda/lib64"
ENV LIBRARY_PATH="$LIBRARY_PATH:/usr/local/cuda/lib64"
ENV PATH=$PATH:$CUDA_HOME/bin
#####################################CHOOSE YOUR ARCHITECTURE################################################
ENV TCNN_CUDA_ARCHITECTURES="86"
#####################################CHOOSE YOUR ARCHITECTURE################################################

# Set working directory
WORKDIR /opt
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

RUN pip install ninja 
RUN pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch  --no-dependencies

RUN echo "Installing pip packages for project..." \
	&& pip --no-cache-dir install  \
	cmake==${CMAKE_VERSION} \
	opencv-python\
	matplotlib==3.6.2 \
	scipy==1.9.3 \
	tensorboardX==2.5.1 \
	tqdm==4.64.1 \
	typing_extensions==4.4.0 \
	commentjson \
	imageio==2.16.2  \
	scikit-learn  \
	pandas \
	open3d \
	meshio

WORKDIR /