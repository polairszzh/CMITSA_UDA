#!/bin/bash

cd ..

# custom config
DATA=${DATA:-/path/to/dataset}   # root directory holding the datasets; override with  DATA=/your/path bash VisDa17.sh
GPU=${GPU:-0}                    # GPU index; override with  GPU=2 bash VisDa17.sh
export CUDA_VISIBLE_DEVICES=${GPU}
TRAINER=MODEL

DATASET=visda17 # name of the dataset
CFG=model
T=1.0
U=1.0
ENT=1.0
PRINT_FREQ=100
NAME=s2r
TAU=0.5
PIN=0.5
TEXT=True
VIS=True
VB=RN101

for SEED in 1 2 3
do
  DIR=rerun_visda/${VB}/${DATASET}/seed_${SEED}
  python train.py --root ${DATA} --trainer ${TRAINER} --dataset-config-file configs/datasets/${DATASET}.yaml --config-file configs/trainers/${TRAINER}/${CFG}.yaml --output-dir ${DIR} --source-domains synthetic --target-domains real --seed ${SEED} TRAINER.MODEL.T ${T} TRAINER.MODEL.TAU ${TAU} TRAINER.MODEL.U ${U} TRAINER.MODEL.ENT ${ENT} TRAINER.MODEL.PIN ${PIN} TRAIN.PRINT_FREQ ${PRINT_FREQ} TRAINER.MODEL.TEXT ${TEXT} TRAINER.MODEL.VISUAL ${VIS}
done
