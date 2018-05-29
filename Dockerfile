FROM gcr.io/tensorflow/tensorflow:latest-devel-gpu-py3

RUN apt-get clean
RUN apt-get update && apt-get install -y sudo

RUN apt-get update --fix-missing

RUN apt-get install -y python3-dev \
                       python3-numpy \
                       python3-pip


RUN pip3 install --upgrade pip

RUN pip3 install --upgrade luigi \
                           matplotlib \
                           plotly


COPY ./ /root/mental_tasks
ENV PYTHONPATH /root/mental_tasks
WORKDIR /root/mental_tasks

ENV TZ=America/Los_Angeles
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
