#!/bin/bash

IMAGE_NAME=artifact-app
TAG=latest

docker run -it $IMAGE_NAME:$TAG bash
