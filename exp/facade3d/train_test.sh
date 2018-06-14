#!/bin/bash

# enter environment if using conda
# source activate caffe

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

if [ -z $EXP_DIR ]; then EXP_DIR=$SCRIPT_DIR; fi
if [ -z $SPLT_CODE ]; then SPLT_CODE="$SCRIPT_DIR/../../splatnet"; fi
if [ -z $SPLT_DATA ]; then SPLT_DATA="$SCRIPT_DIR/../../data"; fi
if [ -z $SKIP_TRAIN ]; then SKIP_TRAIN=0; fi
if [ -z $SKIP_TEST ]; then SKIP_TEST=0; fi

mkdir -p $EXP_DIR

# train
if [ $SKIP_TRAIN -le 0 ]; then
python $SPLT_CODE/semseg3d/train.py $EXP_DIR \
    --arch b64_b128_b128_b128_b64_c64 \
    --lattice x*32_y*32_z*32 x*16_y*16_z*16 x*8_y*8_z*8 x*4_y*4_z*4 x*2_y*2_z*2 \
    --skips 5_1 5_2 5_3 5_4 \
    --feat nx_ny_nz_r_g_b_h \
    --dataset_params jitter_color 0.5 jitter_h 0.001 jitter_rotation 1 root $SPLT_DATA/ruemonge428 \
    --batch_size 4 --sample_size 60000 \
    --base_lr 0.0001 --lr_decay 0.2 --stepsize 8000 --num_iter 16000 --test_interval 100 --snapshot_interval 1000 \
    2>&1 | tee $EXP_DIR/train.log ;
fi

# test & plot
if [ $SKIP_TEST -le 0 ]; then
python $SPLT_CODE/semseg3d/test.py facade \
    --exp_dir $EXP_DIR \
    --gt $SPLT_DATA/ruemonge428/pcl_test.ply \
    --input nx_ny_nz_r_g_b_h_x_y_z \
    --dataset_params subset test ;
fi
