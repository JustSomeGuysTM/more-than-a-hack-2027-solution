#!/usr/bin/env bash
# Overnight jobs on an 80-core CPU pod. Everything logs to ~/logs. Expected: PathCRF 7 folds in
# parallel (~10 cores each) + our GBDT and BiGRU cross-validation.
set -uo pipefail; mkdir -p ~/logs; export OMP_NUM_THREADS=10 MKL_NUM_THREADS=10
cd ~/pathcrf
for k in 0 1 2 3 4 5 6; do
  nohup .venv/bin/python train.py --trial $((200+k)) --lomocv --cv_fold_idx $k --sources sportec --agent_model set_tf --seq_model sab \
    --crf_model dynamic_sparse_crf --macro_type poss_edge --micro_type poss_edge --node_in_dim 8 --target_fps 5 --window_stride 1 --flip_pitch \
    --seq_dim 128 --crf_edge_dim 16 --emit_loss --src_loss --dst_loss --macro_weight 1 --crf_weight 1 --n_epochs 50 --batch_size 32 \
    --start_lr 1e-3 --min_lr 1e-5 --weight_decay 1e-4 > ~/logs/pathcrf_fold$k.log 2>&1 &
done
cd ~/mtah2027
nohup .venv/bin/python src/v2.py --cv > ~/logs/gbdt_cv.log 2>&1 &
nohup .venv/bin/python src/seq.py --cv > ~/logs/seq_cv.log 2>&1 &
wait; echo "all jobs finished" > ~/logs/DONE
