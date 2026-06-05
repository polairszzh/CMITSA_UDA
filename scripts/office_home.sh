#!/bin/bash

cd ..

# custom config
DATA=~/code/uda/dataset # path to dataset
TRAINER=MODEL

DATASET=office_home
CFG=model
T=1.0
TAU=0.6
U=1.0
ENT=1.0
PIN=1.0
TEXT=True
VIS=True
VB=RN50

for SEED in 1 2 3
do
 # Art to Clipart
 NAME=a2c
 DIR=rerun_oh/${VB}/${DATASET}/${NAME}/seed_${SEED}
 CUDA_VISIBLE_DEVICES=1 python train.py --root ${DATA} --trainer ${TRAINER} --dataset-config-file configs/datasets/${DATASET}.yaml --config-file configs/trainers/${TRAINER}/${CFG}.yaml --output-dir ${DIR} --source-domains art --target-domains clipart --seed ${SEED} TRAINER.MODEL.T ${T} TRAINER.MODEL.TAU ${TAU} TRAINER.MODEL.U ${U} TRAINER.MODEL.ENT ${ENT} TRAINER.MODEL.PIN ${PIN} TRAINER.MODEL.TEXT ${TEXT} TRAINER.MODEL.VISUAL ${VIS}

 # Art to Product
 NAME=a2p
 DIR=rerun_oh/${VB}/${DATASET}/${NAME}/seed_${SEED}
 CUDA_VISIBLE_DEVICES=1 python train.py --root ${DATA} --trainer ${TRAINER} --dataset-config-file configs/datasets/${DATASET}.yaml --config-file configs/trainers/${TRAINER}/${CFG}.yaml --output-dir ${DIR} --source-domains art --target-domains product --seed ${SEED} TRAINER.MODEL.T ${T} TRAINER.MODEL.TAU ${TAU} TRAINER.MODEL.U ${U} TRAINER.MODEL.ENT ${ENT} TRAINER.MODEL.PIN ${PIN} TRAINER.MODEL.TEXT ${TEXT} TRAINER.MODEL.VISUAL ${VIS}
 
 # Art to Real World
 NAME=a2r
 DIR=rerun_oh/${VB}/${DATASET}/${NAME}/seed_${SEED}
 CUDA_VISIBLE_DEVICES=1 python train.py --root ${DATA} --trainer ${TRAINER} --dataset-config-file configs/datasets/${DATASET}.yaml --config-file configs/trainers/${TRAINER}/${CFG}.yaml --output-dir ${DIR} --source-domains art --target-domains real_world --seed ${SEED} TRAINER.MODEL.T ${T} TRAINER.MODEL.TAU ${TAU} TRAINER.MODEL.U ${U} TRAINER.MODEL.ENT ${ENT} TRAINER.MODEL.PIN ${PIN} TRAINER.MODEL.TEXT ${TEXT} TRAINER.MODEL.VISUAL ${VIS}

 # Clipart to Art
 NAME=c2a
 DIR=rerun_oh/${VB}/${DATASET}/${NAME}/seed_${SEED}
 CUDA_VISIBLE_DEVICES=1 python train.py --root ${DATA} --trainer ${TRAINER} --dataset-config-file configs/datasets/${DATASET}.yaml --config-file configs/trainers/${TRAINER}/${CFG}.yaml --output-dir ${DIR} --source-domains clipart --target-domains art --seed ${SEED} TRAINER.MODEL.T ${T} TRAINER.MODEL.TAU ${TAU} TRAINER.MODEL.U ${U} TRAINER.MODEL.ENT ${ENT} TRAINER.MODEL.PIN ${PIN} TRAINER.MODEL.TEXT ${TEXT} TRAINER.MODEL.VISUAL ${VIS}

 # Clipart to Product
 NAME=c2p
 DIR=rerun_oh/${VB}/${DATASET}/${NAME}/seed_${SEED}
 CUDA_VISIBLE_DEVICES=1 python train.py --root ${DATA} --trainer ${TRAINER} --dataset-config-file configs/datasets/${DATASET}.yaml --config-file configs/trainers/${TRAINER}/${CFG}.yaml --output-dir ${DIR} --source-domains clipart --target-domains product --seed ${SEED} TRAINER.MODEL.T ${T} TRAINER.MODEL.TAU ${TAU} TRAINER.MODEL.U ${U} TRAINER.MODEL.ENT ${ENT} TRAINER.MODEL.PIN ${PIN} TRAINER.MODEL.TEXT ${TEXT} TRAINER.MODEL.VISUAL ${VIS}

 # Clipart to Real World
 NAME=c2r
 DIR=rerun_oh/${VB}/${DATASET}/${NAME}/seed_${SEED}
 CUDA_VISIBLE_DEVICES=1 python train.py --root ${DATA} --trainer ${TRAINER} --dataset-config-file configs/datasets/${DATASET}.yaml --config-file configs/trainers/${TRAINER}/${CFG}.yaml --output-dir ${DIR} --source-domains clipart --target-domains real_world --seed ${SEED} TRAINER.MODEL.T ${T} TRAINER.MODEL.TAU ${TAU} TRAINER.MODEL.U ${U} TRAINER.MODEL.ENT ${ENT} TRAINER.MODEL.PIN ${PIN} TRAINER.MODEL.TEXT ${TEXT} TRAINER.MODEL.VISUAL ${VIS}

 # Product to Art
 NAME=p2a
 DIR=rerun_oh/${VB}/${DATASET}/${NAME}/seed_${SEED}
 CUDA_VISIBLE_DEVICES=1 python train.py --root ${DATA} --trainer ${TRAINER} --dataset-config-file configs/datasets/${DATASET}.yaml --config-file configs/trainers/${TRAINER}/${CFG}.yaml --output-dir ${DIR} --source-domains product --target-domains art --seed ${SEED} TRAINER.MODEL.T ${T} TRAINER.MODEL.TAU ${TAU} TRAINER.MODEL.U ${U} TRAINER.MODEL.ENT ${ENT} TRAINER.MODEL.PIN ${PIN} TRAINER.MODEL.TEXT ${TEXT} TRAINER.MODEL.VISUAL ${VIS}
 
 # Product to Clipart
 NAME=p2c
 DIR=rerun_oh/${VB}/${DATASET}/${NAME}/seed_${SEED}
 CUDA_VISIBLE_DEVICES=1 python train.py --root ${DATA} --trainer ${TRAINER} --dataset-config-file configs/datasets/${DATASET}.yaml --config-file configs/trainers/${TRAINER}/${CFG}.yaml --output-dir ${DIR} --source-domains product --target-domains clipart --seed ${SEED} TRAINER.MODEL.T ${T} TRAINER.MODEL.TAU ${TAU} TRAINER.MODEL.U ${U} TRAINER.MODEL.ENT ${ENT} TRAINER.MODEL.PIN ${PIN} TRAINER.MODEL.TEXT ${TEXT} TRAINER.MODEL.VISUAL ${VIS}
 
 # Product to Real World
 NAME=p2r
 DIR=rerun_oh/${VB}/${DATASET}/${NAME}/seed_${SEED}
 CUDA_VISIBLE_DEVICES=1 python train.py --root ${DATA} --trainer ${TRAINER} --dataset-config-file configs/datasets/${DATASET}.yaml --config-file configs/trainers/${TRAINER}/${CFG}.yaml --output-dir ${DIR} --source-domains product --target-domains real_world --seed ${SEED} TRAINER.MODEL.T ${T} TRAINER.MODEL.TAU ${TAU} TRAINER.MODEL.U ${U} TRAINER.MODEL.ENT ${ENT} TRAINER.MODEL.PIN ${PIN} TRAINER.MODEL.TEXT ${TEXT} TRAINER.MODEL.VISUAL ${VIS}

 # Real World to Art
 NAME=r2a
 DIR=rerun_oh/${VB}/${DATASET}/${NAME}/seed_${SEED}
 CUDA_VISIBLE_DEVICES=1 python train.py --root ${DATA} --trainer ${TRAINER} --dataset-config-file configs/datasets/${DATASET}.yaml --config-file configs/trainers/${TRAINER}/${CFG}.yaml --output-dir ${DIR} --source-domains real_world --target-domains art --seed ${SEED} TRAINER.MODEL.T ${T} TRAINER.MODEL.TAU ${TAU} TRAINER.MODEL.U ${U} TRAINER.MODEL.ENT ${ENT} TRAINER.MODEL.PIN ${PIN} TRAINER.MODEL.TEXT ${TEXT} TRAINER.MODEL.VISUAL ${VIS}

 # Real World to Clipart
 NAME=r2c
 DIR=rerun_oh/${VB}/${DATASET}/${NAME}/seed_${SEED}
 CUDA_VISIBLE_DEVICES=1 python train.py --root ${DATA} --trainer ${TRAINER} --dataset-config-file configs/datasets/${DATASET}.yaml --config-file configs/trainers/${TRAINER}/${CFG}.yaml --output-dir ${DIR} --source-domains real_world --target-domains clipart --seed ${SEED} TRAINER.MODEL.T ${T} TRAINER.MODEL.TAU ${TAU} TRAINER.MODEL.U ${U} TRAINER.MODEL.ENT ${ENT} TRAINER.MODEL.PIN ${PIN} TRAINER.MODEL.TEXT ${TEXT} TRAINER.MODEL.VISUAL ${VIS}

 # Real World to Product
 NAME=r2p
 DIR=rerun_oh/${VB}/${DATASET}/${NAME}/seed_${SEED}
 CUDA_VISIBLE_DEVICES=1 python train.py --root ${DATA} --trainer ${TRAINER} --dataset-config-file configs/datasets/${DATASET}.yaml --config-file configs/trainers/${TRAINER}/${CFG}.yaml --output-dir ${DIR} --source-domains real_world --target-domains product --seed ${SEED} TRAINER.MODEL.T ${T} TRAINER.MODEL.TAU ${TAU} TRAINER.MODEL.U ${U} TRAINER.MODEL.ENT ${ENT} TRAINER.MODEL.PIN ${PIN} TRAINER.MODEL.TEXT ${TEXT} TRAINER.MODEL.VISUAL ${VIS}
done