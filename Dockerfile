FROM nvidia/cuda:11.5.2-cudnn8-runtime-ubuntu18.04

WORKDIR /home

COPY ./test_detection /home/test_detection

ENV PATH=/home/anaconda3/bin:$PATH

# Install sudo


RUN apt-get update && \
    apt-get install -y sudo && DEBIAN_FRONTEND=noninteractive apt-get install -y net-tools && apt-get install -y  iputils-ping

RUN apt-get -y update && apt-get install -y --no-install-recommends wget \
    && wget https://repo.anaconda.com/archive/Anaconda3-2023.03-1-Linux-x86_64.sh \
    && bash ./Anaconda3-2023.03-1-Linux-x86_64.sh -b -p /home/anaconda3 \
    && rm Anaconda3-2023.03-1-Linux-x86_64.sh \
    #&& cd test_detection && pip install -e . \

RUN conda env create -f detection.yml
#SHELL ["conda", "run", "-n", "detection", "bin/bash", "-c"]
ENV Name detection


RUN pip install av \
    && rm -rf /var/lib/apt/lists/* \