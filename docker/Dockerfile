# FROM ubuntu:23.10
FROM tezzmlops23.azurecr.io/base:v1
# copy the whole code directory
COPY . /digits/
# RUN apt-get update
# RUN apt-get install -y python3 python3-pips
# need python
# no need for conda or venv
WORKDIR /digits
# requirements installation
CMD ["pytest"]