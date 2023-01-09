# syntax=docker/dockerfile:1.4
FROM python:3.9-slim

ENV PYTHONUNBUFFERED=True \
    PORT=${PORT:-9090} \
    PIP_CACHE_DIR=/.cache

WORKDIR /app
COPY requirements.txt .

RUN --mount=type=cache,target=$PIP_CACHE_DIR \
    pip3 install -r requirements.txt

RUN --mount=type=cache,target=$PIP_CACHE_DIR \
    pip install protobuf==3.20.0

COPY . ./

RUN mv libs/object_detection .

EXPOSE $PORT

CMD exec gunicorn --preload --bind :$PORT --workers 1 --threads 8 --timeout 0 _wsgi:app
