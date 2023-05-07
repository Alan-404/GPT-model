FROM python:3.7-slim-buster

WORKDIR /the/workdir/path

COPY . /app

RUN pip install --trusted-host pypi.python.org -r requirements.txt

