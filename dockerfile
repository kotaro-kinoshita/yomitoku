FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

ENV TZ=Asia/Tokyo
ENV DEBIAN_FRONTEND=noninteractive
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get -y update && apt-get -y upgrade
RUN apt-get install -y curl wget unzip vim

RUN apt-get -y install libopencv-dev build-essential clang poppler-utils 

ENV UV_INDEX_STRATEGY="unsafe-best-match"

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

COPY pyproject.toml .

ENV UV_SYSTEM_PYTHON=true \
    UV_COMPILE_BYTECODE=1 \
    UV_CACHE_DIR=/root/.cache/uv \
    UV_LINK_MODE=copy

ENV PATH="/root/.cargo/bin/:$PATH"

RUN --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=.python-version,target=.python-version \
    --mount=type=bind,source=README.md,target=README.md \
    uv sync
RUN . .venv/bin/activate

WORKDIR /workspace