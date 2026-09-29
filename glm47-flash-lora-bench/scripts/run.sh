#!/usr/bin/env bash
# Usage: run.sh <framework> <seq_len> <tag> [extra-json]
# Renders config, runs one training job with the probe, then pushes probe metrics to wandb.
set -uo pipefail
FW=$1; SEQ=$2; TAG=$3; EXTRA=${4:-}
B=/workspace/data/bench
mkdir -p $B/runs/$TAG $B/results
export BENCH_OUT=$B/results/$TAG.probe.jsonl
rm -f $BENCH_OUT
export PYTHONPATH=$B/probe${PYTHONPATH:+:$PYTHONPATH}
export WANDB_PROJECT=glm47-flash-lora-bench WANDB_RUN_GROUP=$FW WANDB_NAME=$TAG WANDB_TAGS="$FW,seq$SEQ"
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1} TOKENIZERS_PARALLELISM=false
CFG=$(/workspace/axolotl-venv/bin/python $B/scripts/make_configs.py $FW $SEQ $TAG ${EXTRA:+"$EXTRA"})
LOG=$B/logs/$TAG.log
echo "[$(date -Is)] start $TAG cfg=$CFG" | tee $LOG
START=$(date +%s)
case $FW in
  axolotl)
    cd /workspace/axolotl && /workspace/axolotl-venv/bin/axolotl train $CFG >> $LOG 2>&1 ;;
  unsloth)
    cd $B/runs/$TAG && $B/unsloth/.venv/bin/python $B/scripts/train_unsloth.py $CFG >> $LOG 2>&1 ;;
  llamafactory)
    cd $B/runs/$TAG && $B/llamafactory/.venv/bin/llamafactory-cli train $CFG >> $LOG 2>&1 ;;
  primerl)
    cd $B/prime-rl/src && LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH} BENCH_DATASET=$B/datasets/synthetic-$SEQ uv run --frozen --no-sync sft @ $CFG >> $LOG 2>&1 ;;
esac
RC=$?
echo "[$(date -Is)] end $TAG rc=$RC wall=$(( $(date +%s) - START ))s" | tee -a $LOG
/workspace/axolotl-venv/bin/python $B/scripts/finalize.py $FW $SEQ $TAG $RC $CFG >> $LOG 2>&1
exit $RC
