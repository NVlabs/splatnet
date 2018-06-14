#!/bin/bash

# enter environment if using conda
# source activate caffe

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

if [ -z $EXP_DIR ]; then EXP_DIR=$SCRIPT_DIR; fi
if [ -z $SPLT_CODE ]; then SPLT_CODE="$SCRIPT_DIR/../../splatnet"; fi
if [ -z $SPLT_DATA ]; then SPLT_DATA="$SCRIPT_DIR/../../data"; fi
if [ -z $SKIP_TRAIN ]; then SKIP_TRAIN=0; fi
if [ -z $SKIP_TEST ]; then SKIP_TEST=0; fi
if [ -z "$CATS" ]; then CATS="laptop car skateboard chair earphone motorbike mug airplane \
pistol lamp bag cap rocket guitar table knife"; fi

mkdir -p $EXP_DIR

# train
if [ $SKIP_TRAIN -le 0 ]; then
for CAT in $CATS; do
python $SPLT_CODE/partseg3d/train.py $EXP_DIR \
    --categories $CAT \
    --arch c32_b64_b128_b256_b256_b256_c128 \
    --skips 6_2 6_3 6_4 6_5 \
    --lattice x*64_y*64_z*64 x*32_y*32_z*32 x*16_y*16_z*16 x*8_y*8_z*8 x*4_y*4_z*4 \
    --feat x_y_z \
    --dataset_params jitter_xyz 0.01 jitter_rotation 10 root $SPLT_DATA/shapenet_ericyi_ply \
    --batch_size 32 --sample_size 3000 \
    --base_lr 0.0001 --lr_decay 0.1 --stepsize 2000 --num_iter 2000 --test_interval 20 --snapshot_interval 100 \
    2>&1 | tee $EXP_DIR/$CAT.log; done ;
fi

# test & plot
if [ $SKIP_TEST -le 0 ]; then
python $SPLT_CODE/partseg3d/test.py shapenet \
    --dataset_params root $SPLT_DATA/shapenet_ericyi_ply \
    --categories $CATS \
    --snapshot best_loss \
    --sample_size -1 --batch_size 1 \
    --exp_dir $EXP_DIR ;
fi
