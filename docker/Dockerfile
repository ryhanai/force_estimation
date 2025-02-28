FROM nvidia/cuda:12.2.0-runtime-ubuntu20.04
# FROM osrf/ros:noetic-desktop-full

# Run a full upgrade and install utilities for development.
ENV DEBIAN_FRONTEND=noninteractive

# Upgrade Packages
RUN apt-get update -q \
    && apt-get upgrade -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Utility packages
RUN apt-get update -q \
    && apt-get install -y software-properties-common \
    && add-apt-repository universe && apt-get update -q \
    && apt-get install -y git build-essential wget curl vim sudo lsb-release locales \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install X11 requirements
RUN apt-get update -q && apt-get install -y \
    iputils-ping telnet x11-apps \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python
RUN apt-get update -q && apt-get install -y \
    python3.8 python3-pip python3-dev python3-tk python-is-python3 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install ROS
ENV ROS_DISTRO=noetic
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu focal main" > /etc/apt/sources.list.d/ros-latest.list'
RUN curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -
RUN apt-get update -q \
    && apt-get install -y ros-noetic-desktop-full \
    && apt-get install -y python3-rosdep python3-rosinstall python3-rosinstall-generator python3-wstool build-essential python3-catkin-tools \
    && rosdep init \
    && rosdep update \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# # Setting for Japanese
# RUN apt-get update -q && apt-get install -y \
#     tzdata locales fonts-noto-cjk fcitx-mozc \
#     && apt-get clean \
#     && rm -rf /var/lib/apt/lists/*
# RUN ln -s -f /usr/share/zoneinfo/Asia/Tokyo /etc/localtime && \
#     dpkg-reconfigure tzdata && \
#     locale-gen ja_JP.UTF-8

# Set user
ARG UID=1000
ARG USER=art
ARG HOME=/home/${USER}
RUN useradd -m -u ${UID} ${USER}

# Create ROS workspace
RUN echo "source /opt/ros/noetic/setup.bash" >> ${HOME}/.bashrc
RUN mkdir -p ${HOME}/catkin_ws/src
RUN cd ${HOME}/catkin_ws && /bin/bash -c "source /opt/ros/noetic/setup.sh; catkin_make"
RUN echo "source ${HOME}/catkin_ws/devel/setup.bash" >> ${HOME}/.bashrc

# Install force estimation
RUN cd ${HOME}/catkin_ws/src && git clone -b airec https://github.com/ryhanai/force_estimation.git
RUN cd ${HOME}/catkin_ws && /bin/bash -c "source /opt/ros/noetic/setup.bash; catkin_make"
# ARG WEIGHT_FILE_ID=1-e002NMWVaFazrGgvLzdr5vN4YDN_w5V
# ARG WEIGHT_FILE_NAME=20250126_1251_33.tgz
ARG WEIGHT_FILE_ID=1drtG3xi0rm91xShf8LuJqF9pDURNSiqb
ARG WEIGHT_FILE_NAME=20250204_1849_08.tgz
RUN mkdir -p ${HOME}/catkin_ws/src/force_estimation/runs \
    && cd ${HOME}/catkin_ws/src/force_estimation/runs \
    && curl -L "https://drive.usercontent.google.com/download?id=${WEIGHT_FILE_ID}&export=download&confirm=t" -o "${WEIGHT_FILE_NAME}" \
    && tar xvfz ${WEIGHT_FILE_NAME} \
    && rm ${WEIGHT_FILE_NAME} \
    && chown -R ${USER}:${USER} ${HOME}/catkin_ws/src/force_estimation
RUN pip install -U pip
RUN cd ${HOME}/catkin_ws/src/force_estimation && pip install -r requirements.txt

# Install DepthAnythingV2
ENV PYTHONPATH="${PYTHONPATH}:${HOME}/Depth-Anything-V2"
RUN cd ${HOME} \
    && git clone https://github.com/DepthAnything/Depth-Anything-V2.git \
    && cd ${HOME}/Depth-Anything-V2 \
    && pip install -r requirements.txt


# How to build docker image
# $ docker build --network=host -t force_estimation .
# How to save docker image into a file
# $ docker save force_estimation:latest | gzip > force_estimation_latest.tar.gz
