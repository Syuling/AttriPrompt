#!/bin/bash

#cd ../..

# custom config
DATA=./data # data root
TRAINER=MaPLe

DATASET=$1
# SEED=$2

CFG=vit_b16_c2_ep5_batch4_2ctx
SHOTS=-1

for SHOTS in -1 
do
    for DATASET in 'imagenet_sketch' 'imagenetv2'
    do
        for SEED in 1 2 3 4 5 6
        do
            DIR=output/64/imagenet/${TRAINER}/${CFG}_${SHOTS}shots/seed${SEED}
            # if [ -d "$DIR" ]; then
            # echo "Results are available in ${DIR}."
            # else
            #     echo "Run this job and save the output to ${DIR}"

            python train.py \
            --root ${DATA} \
            --seed ${SEED} \
            --trainer ${TRAINER} \
            --dataset-config-file configs/datasets/${DATASET}.yaml \
            --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
            --output-dir ${DIR} \
            DATASET.NUM_SHOTS ${SHOTS}
            # fi
        done
    done
done
