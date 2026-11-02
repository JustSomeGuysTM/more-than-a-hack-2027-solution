# SoccerNet ball detector — Gurashish Bagga, with Elijah Cavan (Eli)

**Owner:** Gurashish Bagga. Help: Elijah Cavan (Eli), for the first detector and bucket access.

## What SoccerNet Tracking 2023 is, and why it cannot join the PathCRF / BiGRU training set

- Short broadcast clips (30 s, 25 fps, 1920x1080) from Swiss Super League games. Every frame has human-drawn boxes on
  players, goalkeepers, referees and the ball. MOT format: `<clip>/gt/gt.txt` (frame, track id, x, y, w, h, ...) and a
  `gameinfo.ini` that names each track; the ball's track is labelled `ball`. Verify this after unzipping.
- Our possession models read **pitch coordinates of all 22 players over whole matches, plus per-frame possession
  labels**. SoccerNet has pixel boxes from a moving, zoomed broadcast camera that shows part of the pitch, 30 seconds at a
  time, and no possession or ball-in-play labels. There is nothing in it for PathCRF or the BiGRU to learn from.
- It is exactly what a **ball detector** needs: human-drawn ball boxes on other games. Allowed by the rules; it goes in
  the tech doc's training-data table (source, labelled by SoccerNet's annotators, number of frames).

## What was already tried (`eda/NOTES.md`: "Ball detector", "Ball fusion result", "Detector-noise simulator")

- YOLOv8n on 2,269 tiles cut from our tactical video. Labels: the tracker's (ATD) ball projected through the homography,
  snapped to the brightest blob. Algorithmic, so allowed. Val precision 0.76, recall 0.52.
- Gross errors: at confidence 0.25-0.4, 42% of detections were more than 5 m from the ATD ball; 12% even after the
  temporal-consistency filter (`src/ball_fuse.py`).
- Filling the ATD's gaps with detections broke the dead-ball class: fills on dead frames turned D into H/A. After the
  dead-ball fix it looked better on the public window (0.622 -> 0.664), but the honest check said otherwise.
  `src/external.simulate_detector` adds a detector with our error profile to the degraded DFL ball, and on DFL
  leave-one-out it gave +0.006 at best. Dropped from the submission.
- Weak points: (1) labels only exist where the ATD already sees the ball, so the detector never learns the hard cases
  (occlusion, crowded boxes, far side); (2) one frame at a time, so a sock or a line mark looks like a ball; (3) no human
  labels anywhere.

## Why this probably won't move the Phase 1 score

The submission's home/away calls come from PathCRF, which reads player movement only. The ball reaches the submission
through one path: the BiGRU's dead-ball decisions (stoppage onsets are still ~3 s late). So the payoff of a better ball is
whatever better ball features do for dead balls. **Measure that ceiling before building anything** (step 1). If it is
small, this is the innovation-round piece: a clean, documented use of public human labels with an honest evaluation.

## Steps

1. **Ceiling check** (CPU, about an hour; do this first). Run the DFL leave-one-out with `FILLS=1`
   (`eda/dead_diag.py`), but give `simulate_detector` the parameters of a good detector instead of ours: fill 50% and 80%
   of hidden-ball frames, 2% gross errors, no fills during dead stretches. Compare with the 0.532 / 0.893 baseline. If
   even the 80% case gains less than ~0.02, steps 3-5 are for the innovation round only.
2. **Data** (laptop).
   - `aws s3 cp s3://more-than-a-hack-2027/external/soccernet/SN-Tracking-2023/train.zip . --profile mtah`; unzip
     outside the repo.
   - Pull the ball boxes per clip from `gt.txt`, using the `gameinfo.ini` id of the ball. Count frames with a ball box
     and the ball's pixel-size distribution.
   - Measure the same distribution on our tactical feed (the ATD-projected tiles from `src/ball_detect.py --extract`).
     The tactical camera is wide; broadcast is zoomed, so its ball is bigger. Rescale SoccerNet frames so the ball has
     our pixel size (random scale over our range). This domain gap is the main risk.
3. **Detector.** Start from stock COCO `yolov8n.pt` (it has a sports-ball class), not `src/ball_yolov8n.pt`: that one was
   trained on our match's frames and is out of the public repo. Train on SoccerNet ball tiles (every 5th frame is enough;
   neighbouring frames are near-duplicates) plus our ATD-projected tiles. Hold out ~10 SoccerNet clips for honest
   precision, recall and localisation error. YOLOv8n trains on a laptop (MPS) or Colab; no rented GPU needed.
4. **Temporal layer — where a GRU fits.** Detection stays per frame; the gain comes from linking frames.
   - Keep the top-k candidates per frame at low confidence, not just the best one.
   - Link them through time with a constant-velocity Kalman filter and a Viterbi (or multiple-hypothesis) search over
     candidates. The ball moves smoothly except at touches and rarely exceeds ~40 m/s. This replaces
     `ball_fuse.support`.
   - Learned version: a small GRU over each candidate sequence (position, confidence, speed, distance to the nearest
     player) that scores "this candidate is the ball", trained on SoccerNet clips where the truth is known.
   - If YOLO's recall on tiny balls stays poor: a TrackNet-style heatmap network fed 3 stacked consecutive frames (the
     standard approach for tennis and badminton balls).
5. **Judge it honestly.** On held-out SoccerNet clips (human truth). On our match only against the ATD where it exists
   (easy frames, so optimistic), and never tuned on the public five minutes. Then put the measured error profile into
   `simulate_detector` and rerun step 1 with real numbers. Only a DFL-CV gain justifies a pipeline change, and a change
   also has to keep the CSV reproducible (the current detector pass takes 2.5 h).
6. **Rules.** SoccerNet: public, other games, human labels, allowed and documented. Nothing labelled by a person on our
   match. Weights trained on our match's frames stay in the bucket (`derived/`), never in the public repo.

## Day one (a concrete start, ~half a day)

1. `aws s3 cp s3://more-than-a-hack-2027/external/soccernet/SN-Tracking-2023/train.zip ~/sn/ --profile mtah`, unzip
   outside the repo, open one clip: `gameinfo.ini` lists track ids and names; find the ball's id, draw its boxes from
   `gt/gt.txt` on 20 frames with OpenCV and look at them. Confirms the format before anything else.
2. Histogram of ball box sizes (px) over all clips, and the same on our tactical feed from `src/ball_detect.py
   --extract` tiles. That one plot sizes the domain gap.
3. **Zero-shot baseline in an hour:** stock COCO `yolov8n.pt`, class 32 ("sports ball"), on (a) 200 SoccerNet frames
   with known boxes and (b) our every-50th-frame set from `src/ball_detect.py`, scored against the ATD ball exactly as
   the NOTES table does (hit rate, median error, detections on ATD-missing frames). Every later model must beat it.
4. Only then fine-tune (step 3 of the plan above).

## Supplementary data (all about other games; check each licence before use and list it in the tech doc)

| source | what | note |
|---|---|---|
| SoccerNet Tracking 2023 (in the bucket) | ball + player boxes, 25 fps broadcast clips | main source |
| ISSIA-CNR Soccer dataset | 6 fixed wide cameras with ball annotations | closest to our wide tactical view; academic use |
| Roboflow Universe "football players detection" sets | frames with a `ball` class | many are CC BY 4.0; quality varies |
| SoccerNet Game State Reconstruction 2024 | broadcast clips with pitch-calibrated players | useful for the homography / 2D side, not the ball |

## Where to run what

- SoccerNet-only training can run anywhere (laptop MPS, Colab, the vast.ai box): public data, other games.
- Anything that touches our match's frames (the ATD tiles, inference on BAR-ATH) stays on a teammate's laptop.
- Weights trained with our match's tiles go to `s3://more-than-a-hack-2027/derived/`, never the public repo.

## Ideas worth a slide even if Phase 1 does not use them

- Ball tracking as a filter: Kalman + Viterbi over top-k candidates (step 4). Show a clip of raw detections vs the
  tracked ball.
- Possession from the ball: once a ball track exists, "nearest player to the tracked ball" is a possession signal
  PathCRF does not have; that is the innovation-round story.

## Deliverables

- A `--soccernet` mode in `src/ball_detect.py` (or a new script); weights in the bucket.
- A NOTES section: the ceiling check, SoccerNet held-out precision/recall/error, the DFL-CV result.
- One slide for the innovation round, whatever Phase 1 says.
