FROM node:22-slim

# Set ENV variables
ENV NODE_ENV=production
ENV WORKSPACE_DIR=/workspace

ARG DEBIAN_FRONTEND=noninteractive
RUN mkdir -p /etc/apt && touch /etc/apt/sources.list
RUN sed -i 's|http://archive.ubuntu.com/ubuntu|http://mirrors.aliyun.com/ubuntu|g' /etc/apt/sources.list && \
    sed -i 's|http://security.ubuntu.com/ubuntu|http://mirrors.aliyun.com/ubuntu|g' /etc/apt/sources.list

RUN apt-get update && apt-get install -y \
    wget \
    bzip2 \
    ca-certificates \
    git \
    gcc \
    python3-dev \
    unzip \
    && rm -rf /var/lib/apt/lists/*

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh -O /tmp/miniconda.sh && \
    chmod +x /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh

# Optionally, add conda to PATH
ENV PATH="/opt/conda/bin:${PATH}"

# Initialize conda for shell interaction (optional)
RUN /opt/conda/bin/conda init bash

RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

WORKDIR /agentscope_runtime

RUN conda create -n bfcl python=3.11.13 -y

COPY src/agentscope_runtime/sandbox/box/training_box/ ./training_box/
RUN apt-get install gcc python3-dev -y

RUN git clone https://github.com/ShishirPatil/gorilla.git
RUN conda run -n bfcl pip install -r ./gorilla/berkeley-function-call-leaderboard/.
RUN pip install -r ./training_box/environments/bfcl/requirements.txt

RUN cp -r ./gorilla/berkeley-function-call-leaderboard/bfcl_eval/data ./training_box/bfcl

RUN cd training_box

RUN python ./environments/bfcl/bfcl_dataprocess.py

ENV PYTHONPATH=/agentscope_runtime:$PYTHONPATH
ENV ENV_PATH=/agentscope_runtime/training_box
ENV BFCL_DATA_PATH=$ENV_PATH/bfcl/multiturn_dataset/multi_turn_base_processed.jsonl
ENV BFCL_SPLID_ID_PATH=$ENV_PATH/bfcl/multiturn_dataset/multi_turn_base_split_ids.json

ENV OPENAI_API_KEY=$OPENAI_API_KEY

WORKDIR /agentscope_runtime

CMD . /opt/conda/etc/profile.d/conda.sh && \
    conda activate bfcl && \
    cd /agentscope_runtime/training_box && \
    exec bash bfcl.sh
