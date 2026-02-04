#!/bin/bash

# custom config
DATA=./data
TRAINER=CoOp

#DATASET=$1
CFG=vit_b16_ep200  # config file
CTP=end  # class token position (end or middle)
NCTX=16  # number of context tokens
#SHOTS=$5  # number of shots (1, 2, 4, 8, 16)
CSC=False  # class-specific context (False or True)
HEAD=16
for DATASET in fgvc_aircraft eurosat dtd oxford_flowers oxford_pets stanford_cars caltech101 ucf101 sun397 food101
do
    for SHOTS in -1
    do
        for SEED in 1 2 3 4 5
        do
            DIR=output/a/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}/seed${SEED}
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
            TRAINER.COOP.N_CTX ${NCTX} \
            TRAINER.COOP.CSC ${CSC} \
            TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} \
            DATASET.NUM_SHOTS ${SHOTS}
            fi
        done
    done
done