#!/bin/bash

# custom config
DATA=./data # data root
TRAINER=TaskRes

DATASET=$1
CFG=$2      # config file
ENHANCE=$3  # path to enhanced base weights
SHOTS=-1    # number of shots (1, 2, 4, 8, 16)
SCALE=$4    # scaling factor
HEAD=16

for SHOTS in -1 
do
    for SEED in 1 2 3 4 5
    do
        DIR=output/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/seed${SEED}
        if [ -d "$DIR" ]; then
            echo "Oops! The results exist at ${DIR} (so skip this job)"
        else
        python train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --Head ${HEAD} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
        --output-dir ${DIR} \
        --enhanced-base ${ENHANCE} \
        DATASET.NUM_SHOTS ${SHOTS} \
        TRAINER.TaskRes.RESIDUAL_SCALE ${SCALE}
        fi
    done
done