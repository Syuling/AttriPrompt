#!/bin/bash

# custom config
DATA=dataroot
TRAINER=AttriPrompt


CTP="end"  # class token position (end or middle)
NCTX=2  # number of context tokens
CSC=True  # class-specific context (False or True)
NUM=5
NUM_PROMPT=10


for SHOTS in -1
do
    for DATASET in fgvc_aircraft stanford_cars oxford_flowers oxford_pets 
    do
        for SEED in 1 2 3
        do
            DIR=output/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}/seed${SEED}
            if [ -d "$DIR" ]; then
                echo "Oops! The results exist at ${DIR} (so skip this job)"
            else
            python train.py \
            --root ${DATA} \
            --seed ${SEED} \
            --trainer ${TRAINER} \
            --dataset-config-file configs/datasets/${DATASET}.yaml \
            --config-file configs/trainers/${TRAINER}/${DATASET}.yaml \
            --output-dir ${DIR} \
            TRAINER.ATTRIPROMPT.N_CTX ${NCTX} \
            TRAINER.ATTRIPROMPT.CSC ${CSC} \
            TRAINER.ATTRIPROMPT.CLASS_TOKEN_POSITION ${CTP} \
            MODEL.TOP_K ${NUM} \
            MODEL.NUM_PROMPT ${NUM_PROMPT}\
            DATASET.NUM_SHOTS ${SHOTS}
            # fi
        done
    done
done
