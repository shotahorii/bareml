# Machine Learning From Scratch

Machine learning implementation from scratch. 

## Getting Started

Only prerequisites is numpy installed.

### Example Environment

#### Prerequisites
- docker installed 
- jupyter/minimal-notebook image is pulled by `docker pull jupyter/scipy-notebook`

#### Run 

`cd` to the directory where README.md is located, then type below. 

```
docker run --rm -p 8888:8888 -v "$PWD":/home/jovyan/work jupyter/scipy-notebook
```

Then run notebooks in samples folder.
