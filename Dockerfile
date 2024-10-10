FROM pytorch/pytorch:latest

USER root

RUN --mount=type=cache,target=/.cache/pip pip install --upgrade pip
RUN --mount=type=cache,target=/.cache/pip pip install ipython ipdb

RUN mkdir -p /app

COPY requirements.txt /app/

RUN --mount=type=cache,target=/.cache/pip cd /app && pip install -r requirements.txt

COPY src/ /app/src/
COPY pyproject.toml setup.py README.md /app/

RUN --mount=type=cache,target=/.cache/pip cd /app && pip install '.[dev,test]'

ENV PYTHONUNBUFFERED=1

WORKDIR /app
