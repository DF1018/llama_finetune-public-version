FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/usr/local/cuda-11.7/bin${PATH:+:${PATH}}"
ENV LD_LIBRARY_PATH="/usr/local/cuda-11.7/targets/x86_64-linux/lib:${LD_LIBRARY_PATH}"
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

# 安装 Python 3.10
RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.10 python3.10-distutils python3.10-venv && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN python3.10 -m ensurepip && \
    python3.10 -m pip install --upgrade pip

RUN python3.10 -m pip install --upgrade pip setuptools

RUN pip install lit


#RUN pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117 #df_llama2

RUN apt-get update && apt-get upgrade -y
RUN apt-get install -y wget curl git ffmpeg locales unzip vim sudo cmake sqlite3
RUN apt-get clean && rm -rf /var/lib/apt/lists/*
RUN apt-get update && apt-get upgrade -y
RUN apt-get install -y build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev libsqlite3-dev


# 設定工作目錄
WORKDIR /langchain

# 複製 requirements.txt 到容器中
COPY ./requirements.txt ./

RUN dir

RUN python3.10 -m pip install --upgrade pip && \
    pip install --extra-index-url https://download.pytorch.org/whl/test llama-recipes && \
    pip install -r requirements.txt && \
    pip install accelerate && \
    pip install unstructured && \
    pip install sentence-transformers && \
    pip install chromadb && \
    pip install pysqlite3-binary 
  

  
RUN pip install git+https://github.com/huggingface/transformers

COPY . /app/

# 設定工作目錄
WORKDIR /app/
