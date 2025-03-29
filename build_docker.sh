#!/bin/bash

IMAGE_NAME=artifact-app
TAG=latest

docker build -t $IMAGE_NAME:$TAG .
