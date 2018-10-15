FROM nvidia/cuda:8.0-cudnn6-devel-ubuntu16.04
LABEL maintainer caffe-maint@googlegroups.com

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        ca-certificates \
        wget \
        rsync \
        vim \
        libgl1-mesa-glx \
        libgtk2.0-0 \
        libatlas-base-dev \
        libboost-all-dev \
        libgflags-dev \
        libgoogle-glog-dev \
        libleveldb-dev \
        liblmdb-dev \
        libprotobuf-dev \
        libsnappy-dev \
        libpng12-dev \
        protobuf-compiler
#         protobuf-compiler && \
#     rm -rf /var/lib/apt/lists/*

RUN wget --quiet https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && rm ~/miniconda.sh

ENV CAFFE_ROOT=/opt/caffe
WORKDIR $CAFFE_ROOT

# FIXME: use ARG instead of ENV once DockerHub supports this
# https://github.com/docker/hub-feedback/issues/460
ENV CLONE_TAG=1.0
ENV CONDA=/opt/conda/bin/conda

RUN $CONDA update --yes conda && \
    $CONDA create -n caffe -y python=3.5 && \
    $CONDA install -n caffe -c menpo -y opencv3 && \
    $CONDA install -n caffe -y cython scikit-image scikit-learn matplotlib bokeh ipython h5py nose pandas pyyaml jupyter

RUN git clone -b ${CLONE_TAG} --depth 1 https://github.com/BVLC/caffe.git . && \
    git clone --depth 1 https://github.com/suhangpro/caffe.git caffe-patch && \
    rsync -av caffe-patch/bpcn/ ./ && \
    rm -rf caffe-patch

RUN /bin/bash -c "source /opt/conda/bin/activate caffe && mkdir build && cd build && \
    cmake -DUSE_CUDNN=1 -Dpython_version=3 .. && \
    make -j$(nproc)"

WORKDIR $CAFFE_ROOT/python

RUN /bin/bash -c 'source /opt/conda/bin/activate caffe && \
    sed -i -e "s/python-dateutil>=1.4,<2/python-dateutil>=2.0/g" requirements.txt && \
    for req in $(cat requirements.txt); do pip install $req; done'

ENV PYCAFFE_ROOT=$CAFFE_ROOT/python
ENV PYTHONPATH=$PYCAFFE_ROOT:$PYTHONPATH
ENV PATH=$CAFFE_ROOT/build/tools:$PYCAFFE_ROOT:/opt/conda/bin:$PATH

RUN echo "$CAFFE_ROOT/build/lib" >> /etc/ld.so.conf.d/caffe.conf && ldconfig

WORKDIR /workspace
