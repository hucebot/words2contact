#!/bin/bash
docker run --rm \
    --runtime=nvidia \
    --gpus=all \
    -v /tmp/docker_share:/tmp/docker_share \
    -v $(pwd)/words2contact/prompts/:/home/words2contact/prompts/ \
    -v $(pwd)/words2contact/grammar/:/home/words2contact/grammar/ \
    -v $(pwd)/words2contact/scripts/:/home/words2contact/scripts/ \
    -v $(pwd)/config/:/home/words2contact/config/ \
    -v $(pwd)/submodules/CLIP_Surgery/:/home/words2contact/scripts/CLIP_Surgery \
    -v $(pwd)/data/:/home/words2contact/data/ \
    --ipc host \
    --device /dev/bus/usb \
    --privileged \
    --ulimit rtprio=99 \
    --net host \
    --name words2contact \
    -ti words2contact:latest
