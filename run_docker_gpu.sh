#!/bin/bash

IMAGE_NAME=artifact-app
TAG=latest

docker run --gpus all --shm-size=8g -it $IMAGE_NAME:$TAG bash
