#!/bin/bash -e
. config

USERNAME=art
EXEC_SHELL=bash

docker exec -it \
    -e DISPLAY=unix${DISPLAY} \
    -e NVIDIA_DRIVER_CAPABILITIES=all \
    -u ${USERNAME} \
    ${CONTAINER_NAME} ${EXEC_SHELL}
