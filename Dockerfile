FROM jupyter/scipy-notebook

# Install Chrome for Selenium
# https://stackoverflow.com/questions/51515137/using-selenium-in-python-with-headless-chrome-from-a-docker-container

USER root

RUN ln -snf /bin/bash /bin/sh

RUN sudo apt-get update
RUN wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
RUN dpkg -i google-chrome-stable_current_amd64.deb; apt-get -fy install

RUN wget https://chromedriver.storage.googleapis.com/83.0.4103.39/chromedriver_linux64.zip
RUN unzip chromedriver_linux64.zip chromedriver

# Install any additional packages via pip

USER jovyan

RUN pip install --upgrade pip
RUN pip install selenium