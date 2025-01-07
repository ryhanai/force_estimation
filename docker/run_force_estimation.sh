#!/bin/bash

xhost +local:
. config

IMAGE_NAME=force_estimation
IMAGE_TAG=latest
IMAGE=${IMAGE_NAME}:${IMAGE_TAG}

# user in DOCKER:  art uid:1000 gid:1000
USERNAME=art

echo "CONTAINER_NAME: ${CONTAINER_NAME}"

docker run --gpus all --rm -it --shm-size=3g --ulimit memlock=-1 --ulimit stack=67108864 \
 --network=host \
 --privileged \
 -v /tmp/.X11-unix:/tmp/.X11-unix \
 -e TZ=Asia/Tokyo \
 -e DISPLAY=unix$DISPLAY \
 -e NVIDIA_DRIVER_CAPABILITIES=all \
 -u ${USERNAME} \
 --name ${CONTAINER_NAME} \
 ${IMAGE}
