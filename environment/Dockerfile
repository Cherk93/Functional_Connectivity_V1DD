# hash:sha256:d530de2572cf5ce858ab4e7dc1582a15217f8dce7d9f519546d00de17ec175bd
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
    packaging==23.1 \
    pandas \
    pynwb==3.0.0 \
    scikit-learn \
    scipy \
    seaborn
