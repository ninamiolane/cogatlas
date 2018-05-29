#!/bin/bash -xe
#

docker build -t task_clustering -f Dockerfile .

/usr/bin/nvidia-docker run -i
    -v /Users/nina/code/mental_tasks/data:/Users/nina/code/mental_tasks/data
    task_clustering
