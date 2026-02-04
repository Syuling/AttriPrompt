#!/bin/bash

DATA=./data
TRAINER=LAMM
CFG=vit_b16_ep50_ctxv1 #vit_b16_ep50_ctxv1
HEAD=16
CONT_DIS=False
for SHOTS in -1
do
    for DATASET in fgvc_aircraft eurosat dtd oxford_flowers oxford_pets stanford_cars caltech101 ucf101 imagenet sun397 food101 
    do
        for SEED in  1 2 3 4 5
        do
            DIR=output/${DATASET}/LAMM/16/shots_${SHOTS}/${CFG}/seed${SEED}
            if [ -d "$DIR" ]; then
                echo "Results are available in ${DIR}. Resuming..."
                python train.py \
                --root ${DATA} \
                --seed ${SEED} \
                --Head ${HEAD} \
                --cont_dis ${CONT_DIS} \
                --trainer ${TRAINER} \
                --dataset-config-file configs/datasets/${DATASET}.yaml \
                --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
                --output-dir ${DIR} \
                --triplet-loss \
                --origin-clip \
                DATASET.NUM_SHOTS ${SHOTS} \
                DATASET.SUBSAMPLE_CLASSES all
            else
                echo "Run this job and save the output to ${DIR}"
                python train.py \
                --root ${DATA} \
                --seed ${SEED} \
                --Head ${HEAD} \
                --cont_dis ${CONT_DIS} \
                --trainer ${TRAINER} \
                --dataset-config-file configs/datasets/${DATASET}.yaml \
                --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
                --output-dir ${DIR} \
                --triplet-loss \
                --origin-clip \
                DATASET.NUM_SHOTS ${SHOTS} \
                DATASET.SUBSAMPLE_CLASSES all
            fi
        done
    done      
done  
