#!/usr/bin/env bash
# PathCRF training on a rented GPU (vast.ai / RunPod). Only public data leaves the laptop: DFL open data,
# PFF WC22 and the PathCRF code with our patches. BAR-ATH stays local; inference runs here in 46 s.
#
#   laptop:  bash pod/vast.sh pack                      # -> /tmp/pathcrf_box.tgz (~2 GB)
#            scp -P $PORT /tmp/pathcrf_box.tgz root@$HOST:~/
#            ssh -p $PORT root@$HOST 'tar xzf pathcrf_box.tgz && bash pathcrf/vast.sh setup'
#            ssh -p $PORT root@$HOST 'cd pathcrf && nohup bash vast.sh folds 0 1 2 3 4 5 6 > folds.log 2>&1 &'   # DFL leave-one-out, DFL only (trials 200+k)
#            ssh -p $PORT root@$HOST 'cd pathcrf && SOURCES="sportec pff" nohup bash vast.sh folds 0 1 2 3 4 5 6 > folds_pff.log 2>&1 &'   # same with PFF in training (trials 300+k)
#            ssh -p $PORT root@$HOST 'cd pathcrf && nohup bash vast.sh full > full.log 2>&1 &'                    # final model on all 7 DFL + 64 PFF (trial 400)
#            scp -P $PORT -r root@$HOST:~/pathcrf/saved/[234]* ~/pathcrf/saved/                                    # weights back; run src/pathcrf_run.py --trial 400
set -euo pipefail; [ -d /venv/main ] && export PATH=/venv/main/bin:$PATH   # vast pytorch template venv
case "${1:-}" in
pack)
  cd ~ && tar czf /tmp/pathcrf_box.tgz --exclude=pathcrf/.venv --exclude=pathcrf/data/sportec/tracking --exclude=pathcrf/data/sportec/event --exclude=pathcrf/data/sportec/metadata \
    --exclude=pathcrf/data/pff --exclude='pathcrf/saved/1*' --exclude='pathcrf/saved/2*' --exclude=pathcrf/__pycache__ \
    pathcrf mtah2027/src mtah2027/pod/vast.sh mtah2027/data_raw/external/pff_*.npz
  cp mtah2027/pod/vast.sh pathcrf/vast.sh 2>/dev/null || true; ls -la /tmp/pathcrf_box.tgz ;;
setup)   # on the box; assumes a CUDA torch image (vast "pytorch" template)
  cd ~/pathcrf && cp ~/mtah2027/pod/vast.sh vast.sh
  pip install -q numpy pandas tqdm scipy scikit-learn matplotlib rdp pyarrow torch_geometric
  python -c "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
  # PFF parquets from the npz caches (PathCRF 'pff' source); episodes = alive runs >= 10 s
  mkdir -p data/pff/tracking_processed && cd ~/mtah2027 && PYTHONPATH=~/mtah2027 python -c "
import sys, os, glob; sys.path.insert(0, '.'); from src.pff import pathcrf, CACHE
gs = sorted(os.path.basename(f)[4:-4] for f in glob.glob(CACHE + '/pff_*.npz'))   # npz caches travel in the tarball; raw PFF stays on the laptop
from multiprocessing import Pool
with Pool(8) as pool:
    for r in pool.imap_unordered(pathcrf, gs): print(r, flush=True)"
  ls ~/pathcrf/data/pff/tracking_processed | wc -l; echo "setup done" ;;
folds)   # DFL leave-one-match-out; SOURCES="sportec" (trials 200+k) or "sportec pff" (trials 300+k)
  shift; cd ~/pathcrf; SOURCES=${SOURCES:-sportec}; base=$([ "$SOURCES" = sportec ] && echo 200 || echo 300)
  for k in "$@"; do
    python train.py --trial $((base+k)) --lomocv --cv_fold_idx $k --sources $SOURCES --agent_model set_tf --seq_model sab \
      --crf_model dynamic_sparse_crf --macro_type poss_edge --micro_type poss_edge --node_in_dim 8 --target_fps 5 --window_stride 1 --flip_pitch \
      --seq_dim 128 --crf_edge_dim 16 --emit_loss --src_loss --dst_loss --macro_weight 1 --crf_weight 1 --n_epochs ${N_EPOCHS:-30} --batch_size 64 \
      --start_lr 1e-3 --min_lr 1e-5 --weight_decay 1e-4 --num_workers 2 2>&1 | grep -v -i warn > fold${base}_$k.log
  done; echo "folds done" ;;
full)    # final model: 6 DFL + 63 PFF train, 1 DFL + 1 PFF val -> saved/400
  cd ~/pathcrf && python train.py --trial 400 --sources sportec pff --agent_model set_tf --seq_model sab \
      --crf_model dynamic_sparse_crf --macro_type poss_edge --micro_type poss_edge --node_in_dim 8 --target_fps 5 --window_stride 1 --flip_pitch \
      --seq_dim 128 --crf_edge_dim 16 --emit_loss --src_loss --dst_loss --macro_weight 1 --crf_weight 1 --n_epochs ${N_EPOCHS:-30} --batch_size 64 \
      --start_lr 1e-3 --min_lr 1e-5 --weight_decay 1e-4 --num_workers 2 2>&1 | grep -v -i warn > full.log; echo "full done" ;;
*) echo "usage: vast.sh pack|setup|folds k...|full"; exit 1 ;;
esac
