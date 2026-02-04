#!/bin/bash

# custom config
DATA=./data # data root
TRAINER=ProGrad

DATASET=$1
CFG=vit16_ep100  # config file
CTP=end  # class token position (end or middle)
NCTX=16  # number of context tokens
SHOTS=-1  # number of shots (1, 2, 4, 8, 16)
CSC=False  # class-specific context (False or True)
LAMBDA=1.0

for HEAD in 32 64
do
    for DATASET in 'imagenetv2' 'imagenet_sketch'
    do
        for SEED in 1
        do
            DIR=output/${HEAD}/imagenet/${TRAINER}/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}/seed${SEED}
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
            LOSS.LAMBDA ${LAMBDA} \
            TRAINER.COOP.N_CTX ${NCTX} \
            TRAINER.COOP.CSC ${CSC} \
            TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} \
            DATASET.NUM_SHOTS ${SHOTS}
            # fi
        done
    done
done
