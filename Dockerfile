# runtime env for development 

FROM jupyter/scipy-notebook
ENV DEBIAN_FRONTEND noninteractive

USER root
RUN apt-get update && apt-get install -yq --no-install-recommends \
    graphviz \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

USER $NB_USER
RUN pip install graphviz
RUN pip install torch
RUN pip install torchvision

#COPY requirements.txt ./
#RUN pip install --no-cache-dir -r requirements.txt