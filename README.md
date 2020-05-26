# datascience-code

Code for data science. Not project specific, but general ones which can be used as a template or a sample. 

## Getting Started

### Prerequisites
- docker installed 
- jupyter/scipy-notebook image is pulled by docker pull jupyter/scipy-notebook

### Build a docker image for this repo's environment

`cd` to the directory where Dockerfile is located, then type below. 

```
docker build -t shotahorii/datascience-code .
```

### Run

```
docker run --rm -p 8888:8888 -v "$PWD":/home/jovyan/work shotahorii/datascience-code
```