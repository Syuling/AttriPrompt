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
NUM_PROMPT=5
HEAD=64
CONT_DIS=0  # whether use continuous distribution to sample shots
# BACKBONE="ViT-L/14"
LR=0.001
MAX_EPOCH=50

for HEAD in 16
do
for NUM_PROMPT in 10
do
    for SHOTS in -1
    do
        for DATASET in oxford_pets dtd fgvc_aircraft oxford_flowers stanford_cars caltech101 ucf101 food101 
        # for DATASET in food101
        do
            if [[ ${DATASET} == "imagenet" || ${DATASET} == "sun397" ]]; then
                LR=0.00005
                MAX_EPOCH=30
            else
                LR=0.0002
                MAX_EPOCH=50
            fi
            for SEED in 1 2 3 4 5
            do
                DIR=output/blip/${HEAD}/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}/seed${SEED}
                if [ -d "$DIR" ]; then
                    echo "Oops! The results exist at ${DIR} (so skip this job)"
                else
                python train_blip.py \
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
                DATALOADER.TRAIN_X.HEAD_SIZE ${HEAD}\
                OPTIM.LR ${LR}\
                OPTIM.MAX_EPOCH ${MAX_EPOCH}
                # MODEL.BACKBONE.NAME ${BACKBONE}\
                # 
                # fi
            done
        done
    done
done
done
