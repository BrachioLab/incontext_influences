# Pytorch image from https://hub.docker.com/r/pytorch/pytorch/tags
ARG BASE_IMAGE=pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime
FROM ${BASE_IMAGE} as conda

# Install conda dependencies
RUN conda install -y scikit-learn tqdm
RUN apt-get update && apt-get install -y screen tmux vim git
RUN conda install -y -c huggingface -c conda-forge datasets \
 && pip install accelerate evaluate sentencepiece bitsandbytes \
 && pip install git+https://github.com/huggingface/transformers.git@464d420775653885760e30d24d3703e14f4e8a14