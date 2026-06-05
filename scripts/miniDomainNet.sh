#!/bin/bash

cd ..

# custom config
DATA=~/code/uda/dataset
TRAINER=MODEL

DATASET=minidomainnet
CFG=model
T=1.0
TAU=0.5
U=1.0
ENT=1.0
PRINT_FREQ=100
SEED=1
PIN=0.5
ENT=1.0
VB=RN50
TEXT=True
VIS=True

for SEED in 1 2 3
do
  NAME=c2p
  DIR=rerun_md/${VB}/${DATASET}/${NAME}/seed_${SEED}
  python train.py --root ${DATA} --trainer ${TRAINER} --dataset-config-file configs/datasets/${DATASET}.yaml --config-file configs/trainers/${TRAINER}/${CFG}.yaml --output-dir ${DIR} --source-domains clipart --target-domains painting --seed ${SEED} TRAINER.MODEL.T ${T} TRAINER.MODEL.TAU ${TAU} TRAINER.MODEL.U ${U} TRAINER.MODEL.ENT ${ENT} TRAINER.MODEL.PIN ${PIN} TRAIN.PRINT_FREQ ${PRINT_FREQ} TRAINER.MODEL.TEXT ${TEXT} TRAINER.MODEL.VISUAL ${VIS}

  NAME=c2r
  DIR=rerun_md/${VB}/${DATASET}/${TAU}_${ENT}_${PIN}_${NAME}/seed_${SEED}
  python train.py --root ${DATA} --trainer ${TRAINER} --dataset-config-file configs/datasets/${DATASET}.yaml --config-file configs/trainers/${TRAINER}/${CFG}.yaml --output-dir ${DIR} --source-domains clipart --target-domains real --seed ${SEED} TRAINER.MODEL.T ${T} TRAINER.MODEL.TAU ${TAU} TRAINER.MODEL.U ${U} TRAINER.MODEL.ENT ${ENT} TRAINER.MODEL.PIN ${PIN} TRAIN.PRINT_FREQ ${PRINT_FREQ} TRAINER.MODEL.TEXT ${TEXT} TRAINER.MODEL.VISUAL ${VIS}

  NAME=c2s
  DIR=rerun_md/${VB}/${DATASET}/${TAU}_${ENT}_${PIN}_${NAME}/seed_${SEED}
  python train.py --root ${DATA} --trainer ${TRAINER} --dataset-config-file configs/datasets/${DATASET}.yaml --config-file configs/trainers/${TRAINER}/${CFG}.yaml --output-dir ${DIR} --source-domains clipart --target-domains sketch --seed ${SEED} TRAINER.MODEL.T ${T} TRAINER.MODEL.TAU ${TAU} TRAINER.MODEL.U ${U} TRAINER.MODEL.ENT ${ENT} TRAINER.MODEL.PIN ${PIN} TRAIN.PRINT_FREQ ${PRINT_FREQ} TRAINER.MODEL.TEXT ${TEXT} TRAINER.MODEL.VISUAL ${VIS}

  NAME=p2c
  DIR=rerun_md/${VB}/${DATASET}/${TAU}_${ENT}_${PIN}_${NAME}/seed_${SEED}
  python train.py --root ${DATA} --trainer ${TRAINER} --dataset-config-file configs/datasets/${DATASET}.yaml --config-file configs/trainers/${TRAINER}/${CFG}.yaml --output-dir ${DIR} --source-domains painting --target-domains clipart --seed ${SEED} TRAINER.MODEL.T ${T} TRAINER.MODEL.TAU ${TAU} TRAINER.MODEL.U ${U} TRAINER.MODEL.ENT ${ENT} TRAINER.MODEL.PIN ${PIN} TRAIN.PRINT_FREQ ${PRINT_FREQ} TRAINER.MODEL.TEXT ${TEXT} TRAINER.MODEL.VISUAL ${VIS}

  NAME=p2r
  DIR=rerun_md/${VB}/${DATASET}/${TAU}_${ENT}_${PIN}_${NAME}/seed_${SEED}
  python train.py --root ${DATA} --trainer ${TRAINER} --dataset-config-file configs/datasets/${DATASET}.yaml --config-file configs/trainers/${TRAINER}/${CFG}.yaml --output-dir ${DIR} --source-domains painting --target-domains real --seed ${SEED} TRAINER.MODEL.T ${T} TRAINER.MODEL.TAU ${TAU} TRAINER.MODEL.U ${U} TRAINER.MODEL.ENT ${ENT} TRAINER.MODEL.PIN ${PIN} TRAIN.PRINT_FREQ ${PRINT_FREQ} TRAINER.MODEL.TEXT ${TEXT} TRAINER.MODEL.VISUAL ${VIS}

  NAME=p2s
  DIR=rerun_md/${VB}/${DATASET}/${TAU}_${ENT}_${PIN}_${NAME}/seed_${SEED}
  python train.py --root ${DATA} --trainer ${TRAINER} --dataset-config-file configs/datasets/${DATASET}.yaml --config-file configs/trainers/${TRAINER}/${CFG}.yaml --output-dir ${DIR} --source-domains painting --target-domains sketch --seed ${SEED} TRAINER.MODEL.T ${T} TRAINER.MODEL.TAU ${TAU} TRAINER.MODEL.U ${U} TRAINER.MODEL.ENT ${ENT} TRAINER.MODEL.PIN ${PIN} TRAIN.PRINT_FREQ ${PRINT_FREQ} TRAINER.MODEL.TEXT ${TEXT} TRAINER.MODEL.VISUAL ${VIS}

  NAME=r2c
  DIR=rerun_md/${VB}/${DATASET}/${TAU}_${ENT}_${PIN}_${NAME}/seed_${SEED}
  python train.py --root ${DATA} --trainer ${TRAINER} --dataset-config-file configs/datasets/${DATASET}.yaml --config-file configs/trainers/${TRAINER}/${CFG}.yaml --output-dir ${DIR} --source-domains real --target-domains clipart --seed ${SEED} TRAINER.MODEL.T ${T} TRAINER.MODEL.TAU ${TAU} TRAINER.MODEL.U ${U} TRAINER.MODEL.ENT ${ENT} TRAINER.MODEL.PIN ${PIN} TRAIN.PRINT_FREQ ${PRINT_FREQ} TRAINER.MODEL.TEXT ${TEXT} TRAINER.MODEL.VISUAL ${VIS}

  NAME=r2p
  DIR=rerun_md/${VB}/${DATASET}/${TAU}_${ENT}_${PIN}_${NAME}/seed_${SEED}
  python train.py --root ${DATA} --trainer ${TRAINER} --dataset-config-file configs/datasets/${DATASET}.yaml --config-file configs/trainers/${TRAINER}/${CFG}.yaml --output-dir ${DIR} --source-domains real --target-domains painting --seed ${SEED} TRAINER.MODEL.T ${T} TRAINER.MODEL.TAU ${TAU} TRAINER.MODEL.U ${U} TRAINER.MODEL.ENT ${ENT} TRAINER.MODEL.PIN ${PIN} TRAIN.PRINT_FREQ ${PRINT_FREQ} TRAINER.MODEL.TEXT ${TEXT} TRAINER.MODEL.VISUAL ${VIS}

  NAME=r2s
  DIR=rerun_md/${VB}/${DATASET}/${TAU}_${ENT}_${PIN}_${NAME}/seed_${SEED}
  python train.py --root ${DATA} --trainer ${TRAINER} --dataset-config-file configs/datasets/${DATASET}.yaml --config-file configs/trainers/${TRAINER}/${CFG}.yaml --output-dir ${DIR} --source-domains real --target-domains sketch --seed ${SEED} TRAINER.MODEL.T ${T} TRAINER.MODEL.TAU ${TAU} TRAINER.MODEL.U ${U} TRAINER.MODEL.ENT ${ENT} TRAINER.MODEL.PIN ${PIN} TRAIN.PRINT_FREQ ${PRINT_FREQ} TRAINER.MODEL.TEXT ${TEXT} TRAINER.MODEL.VISUAL ${VIS}

  NAME=s2c
  DIR=rerun_md/${VB}/${DATASET}/${TAU}_${ENT}_${PIN}_${NAME}/seed_${SEED}
  python train.py --root ${DATA} --trainer ${TRAINER} --dataset-config-file configs/datasets/${DATASET}.yaml --config-file configs/trainers/${TRAINER}/${CFG}.yaml --output-dir ${DIR} --source-domains sketch --target-domains clipart --seed ${SEED} TRAINER.MODEL.T ${T} TRAINER.MODEL.TAU ${TAU} TRAINER.MODEL.U ${U} TRAINER.MODEL.ENT ${ENT} TRAINER.MODEL.PIN ${PIN} TRAIN.PRINT_FREQ ${PRINT_FREQ} TRAINER.MODEL.TEXT ${TEXT} TRAINER.MODEL.VISUAL ${VIS}

  NAME=s2p
  DIR=rerun_md/${VB}/${DATASET}/${TAU}_${ENT}_${PIN}_${NAME}/seed_${SEED}
  python train.py --root ${DATA} --trainer ${TRAINER} --dataset-config-file configs/datasets/${DATASET}.yaml --config-file configs/trainers/${TRAINER}/${CFG}.yaml --output-dir ${DIR} --source-domains sketch --target-domains painting --seed ${SEED} TRAINER.MODEL.T ${T} TRAINER.MODEL.TAU ${TAU} TRAINER.MODEL.U ${U} TRAINER.MODEL.ENT ${ENT} TRAINER.MODEL.PIN ${PIN} TRAIN.PRINT_FREQ ${PRINT_FREQ} TRAINER.MODEL.TEXT ${TEXT} TRAINER.MODEL.VISUAL ${VIS}

  NAME=s2r
  DIR=rerun_md/${VB}/${DATASET}/${TAU}_${ENT}_${PIN}_${NAME}/seed_${SEED}
  python train.py --root ${DATA} --trainer ${TRAINER} --dataset-config-file configs/datasets/${DATASET}.yaml --config-file configs/trainers/${TRAINER}/${CFG}.yaml --output-dir ${DIR} --source-domains sketch --target-domains real --seed ${SEED} TRAINER.MODEL.T ${T} TRAINER.MODEL.TAU ${TAU} TRAINER.MODEL.U ${U} TRAINER.MODEL.ENT ${ENT} TRAINER.MODEL.PIN ${PIN} TRAIN.PRINT_FREQ ${PRINT_FREQ} TRAINER.MODEL.TEXT ${TEXT} TRAINER.MODEL.VISUAL ${VIS}
done