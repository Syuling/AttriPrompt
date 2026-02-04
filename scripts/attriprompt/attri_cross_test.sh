#!/bin/bash

# custom config
DATA=./data
TRAINER=AttriPrompt

DATASET=dtd #fgvc_aircraft
CFG=vit_b162  # config file
CTP="end"  # class token position (end or middle)
NCTX=2  # number of context tokens
# SHOTS=$5  # number of shots (1, 2, 4, 8, 16)
CSC=True  # class-specific context (False or True)
NUM=5
NUM_PROMPT=10


for HEAD in 16 32 64
do
for SHOTS in -1
do
    for DATASET in 'imagenetv2' 'imagenet_sketch'
    do
        for SEED in 1 2 3 4 5
        do
            DIR=output/${HEAD}/imagenet/${TRAINER}/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}/seed${SEED}
            python train.py \
            --root ${DATA} \
            --seed ${SEED} \
            --trainer ${TRAINER} \
            --dataset-config-file configs/datasets/${DATASET}.yaml \
            --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
            --output-dir ${DIR} \
            TRAINER.ATTRIPROMPT.N_CTX ${NCTX} \
            TRAINER.ATTRIPROMPT.CSC ${CSC} \
            TRAINER.ATTRIPROMPT.CLASS_TOKEN_POSITION ${CTP} \
            MODEL.TOP_K ${NUM} \
            MODEL.NUM_PROMPT ${NUM_PROMPT}\
            DATASET.NUM_SHOTS ${SHOTS}\
            DATALOADER.TRAIN_X.HEAD_SIZE ${HEAD}
        done
    done
done
done