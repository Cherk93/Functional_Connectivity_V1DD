# hash:sha256:3f51ab12adbddc2000a27f00953d5c9a683ead52e34b32274a3ec8951975d528
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/mambaforge3:23.1.0-4-python3.10.12-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    caveclient \
    hdmf==3.14.6 \
    hdmf-zarr==0.11.3 \
    matplotlib \
    numpy \
    nwbwidgets==0.11.3 \
    packagin==23.1 \
    pandas \
    pynwb==3.0.0 \
    scikit-learn \
    scipy \
    seaborn
