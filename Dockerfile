FROM continuumio/miniconda3 AS base

SHELL ["/bin/bash", "--login", "-c"]

# Some areas require
COPY .devcontainer/src/.condarc /root/.condarc
RUN conda clean -i && conda config --show
RUN python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade pip && \
    pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

RUN mkdir -p /app
WORKDIR /app
COPY environment.yml ./

RUN conda env create -f environment.yml && \
    conda clean -afy && \
    rm -rf /opt/conda/pkgs/*

RUN sed -i 's/conda activate base/conda activate deepblock_env/' ~/.bashrc

ENV ESM_NAME esm2_t6_8M_UR50D
ADD https://dl.fbaipublicfiles.com/fair-esm/models/${ESM_NAME}.pt /root/.cache/torch/hub/checkpoints/${ESM_NAME}.pt
ADD https://dl.fbaipublicfiles.com/fair-esm/regression/${ESM_NAME}-contact-regression.pt /root/.cache/torch/hub/checkpoints/${ESM_NAME}-contact-regression.pt

COPY . .
RUN pip install -e .
