# Adapted from: https://raw.githubusercontent.com/CosmiQ/solaris/master/docker/gpu/Dockerfile
FROM nvidia/cuda:10.1-cudnn7-devel

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get install -y \
	ca-certificates git wget sudo ninja-build zip unzip vim curl

# prep apt-get and cudnn
RUN apt-get update && apt-get install -y --no-install-recommends apt-utils

# install requirements
RUN apt-get update \
  && apt-get install -y --no-install-recommends \
    bc \
    bzip2 \
    ca-certificates \
    curl \
    git \
    libgdal-dev \
    libssl-dev \
    libffi-dev \
    libncurses-dev \
    libgl1 \
    jq \
    nfs-common \
    parallel \
    python-dev \
    python-pip \
    python-wheel \
    python-setuptools \
    unzip \
    vim \
    wget \
    build-essential \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

SHELL ["/bin/bash", "-c"]
ENV PATH /opt/conda/bin:$PATH

# install anaconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -tipsy && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

# install dependencies
RUN pip install --upgrade pip 
RUN pip install tensorboard cmake   # cmake from apt-get is too old
RUN pip install torch==1.9.0 torchvision==0.10.0 -f https://download.pytorch.org/whl/cu101/torch_stable.html
RUN pip install ipython jupyter tqdm matplotlib \
                opencv-python scikit-learn scikit-image \
				albumentations==1.0.0 catalyst==21.5 \
				efficientnet_pytorch==0.7.1 pandas \
				flask waitress

# copy current path to workdir
WORKDIR /work
COPY . /work/
RUN chmod 777 download_train.sh
RUN chmod 777 download_deploy.sh

# open ports for jupyter notebook, http and api
EXPOSE 8081 8099 8989

RUN [ "/bin/bash" ]
