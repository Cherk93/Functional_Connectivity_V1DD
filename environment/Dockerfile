# hash:sha256:ef840b7b2b53e9450ed0f4eb40eb8ed008b9f1bfb5b9a0ac2bb724123b72ba29
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/mambaforge3:23.1.0-4-python3.10.12-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    hdmf==3.14.6 \
    hdmf-zarr==0.11.3 \
    matplotlib==3.10.5 \
    numpy==2.2.6 \
    nwbwidgets==0.11.3 \
    pandas==2.3.2 \
    pynwb==3.0.0
