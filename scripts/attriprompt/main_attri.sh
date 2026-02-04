#!/bin/bash

# custom config
DATA=./data
TRAINER=AttriPrompt

DATASET=dtd #fgvc_aircraft
CFG=vit_b16  # config file
CTP="end"  # class token position (end or middle)
NCTX=2  # number of context tokens
# SHOTS=$5  # number of shots (1, 2, 4, 8, 16)
CSC=True  # class-specific context (False or True)
NUM=5
NUM_PROMPT=10
HEAD=16
CONT_DIS=0  # whether use continuous distribution to sample shots


for HEAD in 16
do
for NCTX in 2
do
    for SHOTS in -1
    do
        for DATASET in fgvc_aircraft stanford_cars dtd oxford_flowers oxford_pets caltech101 ucf101 sun397 food101 imagenet 
        do
            for SEED in 1 2 3 4 5
            do
                DIR=output/${HEAD}/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}/seed${SEED}
                if [ -d "$DIR" ]; then
                    echo "Oops! The results exist at ${DIR} (so skip this job)"
                else
                python train.py \
                --root ${DATA} \
                --seed ${SEED} \
                --Head ${HEAD} \
                --cont_dis ${CONT_DIS} \
                --trainer ${TRAINER} \
                --dataset-config-file configs/datasets/${DATASET}.yaml \
                --config-file configs/trainers/${TRAINER}/${DATASET}.yaml \
                --output-dir ${DIR} \
                TRAINER.ATTRIPROMPT.N_CTX ${NCTX} \
                TRAINER.ATTRIPROMPT.CSC ${CSC} \
                TRAINER.ATTRIPROMPT.CLASS_TOKEN_POSITION ${CTP} \
                MODEL.TOP_K ${NUM} \
                MODEL.NUM_PROMPT ${NUM_PROMPT}\
                DATASET.NUM_SHOTS ${SHOTS}\
                DATALOADER.TRAIN_X.HEAD_SIZE ${HEAD}
                fi
            done
        done
    done
done
done
