#!/bin/bash
docker run --rm \
    --runtime=nvidia \
    --gpus=all \
    -v /tmp/docker_share:/tmp/docker_share \
    -v $(pwd)/words2contact/:/home/words2contact/words2contact/ \
    -v $(pwd)/config/:/home/words2contact/words2contact/config/ \
    -v $(pwd)/submodules/CLIP_Surgery/:/home/words2contact/words2contact/CLIP_Surgery \
    -v $(pwd)/data/:/home/words2contact/data/ \
    -v $(pwd)/main.py:/home/words2contact/main.py \
    -e OPENAI_KEY=$OPENAI_KEY \
    --ipc host \
    --device /dev/bus/usb \
    --privileged \
    --ulimit rtprio=99 \
    --net host \
    --name words2contact \
    -ti words2contact:latest
