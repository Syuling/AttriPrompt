#!/bin/bash

#cd ../..

# custom config
# DATA=/home/suyuling/experience/multi_modal/cross_modal_adaptation-our/data
DATA=./data # data root
TRAINER=CoCoOp

DATASET=imagenet
# SEED=$1

CFG=vit_b16_c4_ep10_batch1_ctxv1
SHOTS=1

for SHOTS in -1
do
    for DATASET in 'imagenetv2' 'imagenet_sketch'
    do
        for SEED in 1 2 3
        do
            DIR=output/imagenet/${TRAINER}/${CFG}_${SHOTS}shots/seed${SEED}
            # if [ -d "$DIR" ]; then
            #     echo "Results are available in ${DIR}. Skip this job"
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


