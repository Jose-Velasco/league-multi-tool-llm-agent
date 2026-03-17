FROM mcr.microsoft.com/devcontainers/python:3.12

# recent breakage from am update to base image
# Remove broken Yarn repo inherited from base image
RUN rm -f /etc/apt/sources.list.d/yarn.list \
          /etc/apt/sources.list.d/yarn.sources

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    build-essential \
    git \
    curl \
    ca-certificates \
  && rm -rf /var/lib/apt/lists/*

# This instruction tells Docker to use the specified GitHub Container Registry image as a source stage. It's an efficient way to access files from another image without building from scratch.
# /uv /uvx: These are the source paths within the source image, specifying the uv and uvx binaries that were built there.
# /usr/local/bin/: This is the destination directory in your current Docker image where the binaries will be copied. This directory is typically already in the system's PATH, making the uv and uvx commands globally accessible to subsequent instructions in your Dockerfile.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /home/dev/src