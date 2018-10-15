### Installing Caffe, bilateralNN, and their dependencies 
Step-by-step installation instructions for Ubuntu 16.04. 
        
* Make sure CUDA and cuDNN are installed
            
    We tested on CUDA8/cuDNN6 and CUDA9/cuDNN7. But Caffe typically supports a wide range of CUDA/cuDNN versions. 

* Install system-wide dependencies
    ```bash
    sudo apt-get update
    sudo apt-get install -y --no-install-recommends \
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
    ```

* Install conda and additional packages
    ```bash
    wget --quiet https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    chmod a+x miniconda.sh
    ./miniconda.sh  # you will be prompted to agree on terms and to configure the installation
    rm miniconda.sh
    source ~/.bashrc
    conda update --yes conda
    conda create -n caffe -y python=3.5
    conda install -n caffe -c menpo -y opencv3
    conda install -n caffe -y cython scikit-image scikit-learn matplotlib bokeh ipython h5py nose pandas pyyaml jupyter
    ```

* Install Caffe and bilateralNN
    ```bash
    CAFFE_ROOT=~/caffe  # specify where caffe will be installed (does not need to be inside the main project)
    git clone -b 1.0 --depth 1 https://github.com/BVLC/caffe.git $CAFFE_ROOT
    cd $CAFFE_ROOT
    git clone --depth 1 https://github.com/suhangpro/caffe.git caffe-patch
    rsync -av caffe-patch/bpcn/ ./
    rm -rf caffe-patch
    source activate caffe
    mkdir build && cd build
    cmake -DUSE_CUDNN=1 -Dpython_version=3 ..
    make -j$(nproc)
    cd $CAFFE_ROOT/python
    sed -i -e "s/python-dateutil>=1.4,<2/python-dateutil>=2.0/g" requirements.txt
    for req in $(cat requirements.txt); do pip install $req; done  # 'msgpack' warning message can be safely ignored
    cd $CAFFE_ROOT
    echo "
    # caffe paths
    export PYCAFFE_ROOT=$(pwd)/python
    export PYTHONPATH=\$PYCAFFE_ROOT:\$PYTHONPATH
    export PATH=\$CAFFE_ROOT/build/tools:\$PYCAFFE_ROOT:\$PATH
    " >> ~/.bashrc  # this puts path in your .bashrc file
    source ~/.bashrc
    sudo /bin/bash -c 'echo "$(pwd)/build/lib" >> /etc/ld.so.conf.d/caffe.conf'
    sudo ldconfig
    ```

* Ready to use!
    ```bash
    # enter conda environment if you are not already in one
    source activate caffe
    ```

Note that for multi-GPU usage (not used or tested in our experiments and demos) you will also need NCCL:
* Before installing Caffe, install NCCL following [official instructions](https://github.com/NVIDIA/nccl). 
* Add `-DUSE_NCCL=1` flag for cmake. 
