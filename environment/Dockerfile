# hash:sha256:80b1ca4a51d0c159ae5e1ac29366a047514ec5bded4c30d5457f93fde60c83fc
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/mambaforge3:23.1.0-4-python3.10.12-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    caveclient==6.5.0 \
    hdmf==3.14.6 \
    hdmf-zarr==0.11.3 \
    matplotlib==3.10.5 \
    numpy==1.26.4 \
    nwbwidgets==0.11.3 \
    packaging==23.1 \
    pandas==2.3.2 \
    pynwb==3.0.0 \
    scikit-learn==1.7.1 \
    scipy==1.15.3 \
    seaborn==0.13.2
