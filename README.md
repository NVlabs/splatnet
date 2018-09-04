## SPLATNet: Sparse Lattice Networks for Point Cloud Processing (CVPR2018)

### License

Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode). 


### Paper
[arXiv](https://arxiv.org/abs/1802.08275)
```
@inproceedings{su18splatnet,
  author={Su, Hang and Jampani, Varun and Sun, Deqing and Maji, Subhransu and Kalogerakis, Evangelos and Yang, Ming-Hsuan and Kautz, Jan},
  title     = {{SPLATN}et: Sparse Lattice Networks for Point Cloud Processing},
  booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages     = {2530--2539},
  year      = {2018}
}
```

### Usage
       
1. Install [Caffe](http://caffe.berkeleyvision.org) and [bilateralNN](https://github.com/MPI-IS/bilateralNN)
    
    Note that our code uses Python3. 
    * Please follow the instructions on the [bilateralNN](https://github.com/MPI-IS/bilateralNN) repo. 
    * A step-by-step installation guide for Ubuntu 16.04 is provided in [INSTALL.md](INSTALL.md). 
    * Alternatively, you can install [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) and use this docker image: 
        ```bash
        docker pull suhangpro/caffe:bpcn
        ``` 
        You can also build your own image with this [Dockerfile](Dockerfile). 

2. Include the project to your python path so imports can be found, e.g. 
    ```bash
    export PYTHONPATH=<PATH_TO_PROJECT_ROOT>:$PYTHONPATH
    ```

3. Download and prepare data files under folder `data/`
    
    See instructions in data/[README.md](data/README.md).
    
4. Usage examples
    * 3D facade segmentation
        * test pre-trained model
            ```bash
            cd exp/facade3d
            ./dl_model_facade3d.sh  # download pre-trained model
            SKIP_TRAIN=1 ./train_test.sh
            ```
            Prediction is output at `pred_test.ply`, with evaluation results in `test.log`.
        * or, train and evaluate
            ```bash
            cd exp/facade3d
            ./train_test.sh
            ```
    * ShapeNet Part segmentation
        * test pre-trained model
            ```bash
            cd exp/shapenet3d
            ./dl_model_shapenet3d.sh  # download pre-trained model
            ./test_only.sh
            ```
            Predictions are under `pred/`, with evaluation results in `test.log`.
        * or, train and evaluate
            ```bash
            cd exp/shapenet3d
            ./train_test.sh
            ```
    * Joint 2D-3D experiments
    
        (coming soon)

### References 
We make extensive use of bilateralNN, which is proposed in these publications:
* V. Jampani, M. Kiefel and P. V. Gehler. Learning Sparse High-Dimensional Filters: Image Filtering, Dense CRFs and Bilateral Neural Networks. CVPR, 2016.
* M.Kiefel, V. Jampani and P. V. Gehler. Permutohedral Lattice CNNs. ICLR Workshops, 2015.
