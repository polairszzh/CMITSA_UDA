#!/bin/bash
#
# Generic launcher: one dataset, one domain pair, any number of seeds.
#
# Fill in the required fields below, then run from this directory:
#     DATA=/path/to/dataset GPU=0 bash run.sh
#
# Every field can also be overridden from the environment, e.g.
#     DATASET=visda17 SRC=synthetic TGT=real BACKBONE=RN101 bash run.sh

# ---------------- required ----------------
DATASET=${DATASET:-}   # office_home | visda17 | minidomainnet
SRC=${SRC:-}           # source domain, e.g. art
TGT=${TGT:-}           # target domain, e.g. clipart

# ---------------- optional ----------------
BACKBONE=${BACKBONE:-} # RN50 | RN101 | ViT-B/16 ; empty = keep the value in configs/datasets/${DATASET}.yaml
TRAINER=${TRAINER:-MODEL}
CFG=${CFG:-model}
SEEDS=${SEEDS:-"1 2 3"}
OUT=${OUT:-output}
DATA=${DATA:-/path/to/dataset}
GPU=${GPU:-0}

# ---------------- hyperparameters (values used in the paper) ----------------
T=1.0       # temperature applied to the logits
TAU=0.5     # confidence threshold gamma
U=1.0       # weight of L_ce^t + L_sc^t
ENT=1.0     # weight of the mutual-information loss
PIN=0.5     # weight of the prompt-guided loss
TEXT=True   # enable VPA
VIS=True    # enable PINA

for v in DATASET SRC TGT; do
  if [ -z "${!v}" ]; then
    echo "[error] ${v} is empty."
    echo "        Set it at the top of this script, or pass it explicitly, e.g."
    echo "        DATASET=office_home SRC=art TGT=clipart bash run.sh"
    exit 1
  fi
done

# the code lives one level above scripts/, as in the other scripts of this folder
cd ..

export CUDA_VISIBLE_DEVICES=${GPU}

OPTS="TRAINER.MODEL.T ${T} TRAINER.MODEL.TAU ${TAU} TRAINER.MODEL.U ${U}"
OPTS="${OPTS} TRAINER.MODEL.ENT ${ENT} TRAINER.MODEL.PIN ${PIN}"
OPTS="${OPTS} TRAINER.MODEL.TEXT ${TEXT} TRAINER.MODEL.VISUAL ${VIS}"
if [ -n "${BACKBONE}" ]; then
  OPTS="${OPTS} MODEL.BACKBONE.NAME ${BACKBONE}"
fi

for SEED in ${SEEDS}; do
  DIR=${OUT}/${BACKBONE:-${DATASET}}/${DATASET}/${SRC}2${TGT}/seed_${SEED}
  echo "[run] ${DATASET}: ${SRC} -> ${TGT}, seed ${SEED}, gpu ${GPU} -> ${DIR}"
  python train.py \
    --root ${DATA} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    --source-domains ${SRC} \
    --target-domains ${TGT} \
    --seed ${SEED} \
    ${OPTS}
done
