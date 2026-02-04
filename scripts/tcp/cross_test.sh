#!/bin/bash

# custom config
DATA=./data
TRAINER=TCP
WEIGHT=1.0

CFG=vit_b16_ep100_ctxv1
CTP=end  # class token position (end or middle)
NCTX=4  # number of context tokens
SHOTS=-1  # number of shots (1, 2, 4, 8, 16)
CSC=False  # class-specific context (False or True)
FOLDER=output

for HEAD in 32 64
do
    for DATASET in 'imagenetv2' 'imagenet_sketch' #oxford_pets dtd fgvc_aircraft eurosat oxford_flowers stanford_cars caltech101 ucf101 food101
    do
        for SEED in 1 2 3 4 5
        do
            DIR=${FOLDER}/${HEAD}/imagenet/${TRAINER}/shots_${SHOTS}_${WEIGHT}_${NCTX}/${CFG}/seed${SEED}
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
            TRAINER.COOP.N_CTX ${NCTX} \
            TRAINER.COOP.CSC ${CSC} \
            TRAINER.COOP.W ${WEIGHT} \
            TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} \
            DATASET.NUM_SHOTS ${SHOTS} \
            DATASET.SUBSAMPLE_CLASSES all
            # fi
        done
    done
done