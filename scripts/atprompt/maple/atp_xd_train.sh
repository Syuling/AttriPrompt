#!/bin/bash

# custom config
DATA=./data
TRAINER=MaPLe_ATP
CFG=vit_l14_c2_ep5_batch4_2ctx
SHOTS=16
CONT_DIS=0

for alpha in 0.0
do
for SHOTS in -1
do
for HEAD in 16
do
for DATASET in dtd caltech101 ucf101 imagenet fgvc_aircraft oxford_flowers oxford_pets stanford_cars food101 sun397
# for DATASET in fgvc_aircraft
# for DATASET in 'imagenetv2' 'imagenet_sketch'
do
for SEED in 1 2 3 4 5
do
        if [[ ${DATASET} == "imagenet" || ${DATASET} == "sun397" || ${DATASET} == "stanford_cars" || ${DATASET} == "food101" || ${DATASET} == "dtd" || ${DATASET} == "imagenetv2" || ${DATASET} == "imagenet_sketch" ]]; then
                NCTX=4
                EPO=5
        else
                NCTX=4
                EPO=10
        fi
        if [[ ${DATASET} == "fgvc_aircraft" || ${DATASET} == "ucf101" ]]; then
                NCTX=2
                EPO=5
        fi

        DIR=output/${DATASET}/${HEAD}/${TRAINER}/${CFG}_${SHOTS}shots_few_shots/nctx${NCTX}_csc${CSC}_ctp${CTP}/CTX_${NCTX}_epo_${EPO}/seed${SEED}

        python train.py \
                --root ${DATA} \
                --seed ${SEED} \
                --Head ${HEAD} \
                --cont_dis ${CONT_DIS} \
                --alpha ${alpha} \
                --trainer ${TRAINER} \
                --dataset-config-file configs/datasets/${DATASET}.yaml \
                --config-file configs/trainers/MaPLe/${CFG}.yaml \
                --output-dir ${DIR} \
                DATASET.NUM_SHOTS ${SHOTS} \
                DATASET.SUBSAMPLE_CLASSES all \
                TRAINER.ATPROMPT.USE_ATPROMPT True \
                TRAINER.ATPROMPT.N_ATT1 ${NCTX} \
                TRAINER.ATPROMPT.N_ATT2 ${NCTX} \
                TRAINER.ATPROMPT.N_ATT3 ${NCTX} \
                OPTIM.MAX_EPOCH ${EPO}\

        
done
done
done
done
done