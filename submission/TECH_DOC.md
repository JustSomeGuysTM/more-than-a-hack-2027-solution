# Ball possession for BAR–ATH — technical document (draft, team <team_name>)

> Status 2026-10-27: draft. Numbers marked **TODO(gpu)** are filled in when the PathCRF training
> run finishes (~2026-10-31). Final candidate choice and the submitted CSV follow from those numbers.

## 1. Approach in one paragraph
Two stages. (1) A per-frame model turns the tactical tracking (players, ball when visible) into
probabilities for *home / away / dead*; (2) a Viterbi decoder with a switch penalty turns those
into whole possession spells, because the metric (spellF1) scores possessions, not frames. On top
of that we run **PathCRF**, a published possession-path model that predicts the ball carrier from
player movement alone (no ball needed), and take its home/away call on alive frames while our own
model decides when the ball is dead. Everything is trained on public data from *other* matches
with real possession labels; nothing is tuned on the 5 public minutes of this game and no hand
labels of this game exist anywhere in the pipeline.

Why this and not something else: the given ball track is missing on 41% of graded frames and the
player tracks are 46 unidentified slots with identity churn. A rule on "nearest player to the
ball" (the organisers' baseline, spellF1 0.24) fails exactly where the ball is missing. A learned
model can read possession from team shape, player speeds and time-since-last-touch, and a sequence
model over player movement (PathCRF) does not need the ball at all. We validated every choice on
held-out external matches with real labels, which is the only honest signal available.

## 2. Pipeline
```
tactical ATD (25 fps, 46 slots, ball 57% visible)
   ├─ src/features.py        46 slot-agnostic per-frame features (per-team nearest distances/counts, ball
   │                          visibility & gap length, ball speed, position, team spread/speed, ±0.5 s/±2 s
   │                          context, seconds since/until last home/away touch, ball off pitch, speed jump)
   ├─ src/v2.py              HistGradientBoosting  -> log P(H/A/D) per frame -> Viterbi (switch 10)
   ├─ src/seq.py             BiGRU (hidden 64, 30-s windows) over the same features -> Viterbi (switch 3)
   ├─ src/pathcrf_io.py      stitch 46 slots into 11 persistent tracks per team, cut alive episodes (≥10 s)
   └─ src/pathcrf_run.py     PathCRF home/away on alive frames + our dead-ball frames -> final CSV
```
Final label per frame = PathCRF team if the frame is alive and covered by an episode, else our
model's label (H / A / D). `X` is never emitted on graded frames.

## 3. Possession rules we implemented
- **Dead ball (D)**: learned from DFL's `ballstatus` code (official alive/dead). DFL freezes the ball
  while dead whereas the BAR–ATH tracker keeps reporting a moving ball during stoppages, so during
  training the DFL dead ball is re-synthesised (carried by the nearest player or drifting) and the model
  has to read stoppages from the players: walking speeds, spread, set-piece shapes, ball off the pitch.
  Dead runs shorter than 3 s are folded into the surrounding possession (threshold from DFL CV).
  Restarts end D at the first touch.
- **Loose balls / challenges**: no explicit rule; possession stays with the last team in
  control until the decoder sees enough evidence of a controlled touch by the other team.
  The switch penalty (chosen by leave-one-match-out CV on DFL, never on this game) makes a change
  of possession cost roughly one second of contrary evidence, so a single deflection does not flip
  the label but a won tackle does.
- **PathCRF** defines possession as a path of carrier→receiver edges between identified players;
  a possession spell is a maximal run of edges within one team. Goalkeeper identity is not given
  on our data; the model is used without it.
- Kick-offs at frames 11185 and 113742 (ball leaving the centre spot); frames before the first
  kick-off and between halves are D.

## 4. Training data (every labelled dataset, none from the provided game)
| dataset | source / licence | labelled by | volume | used for |
|---|---|---|---|---|
| DFL / IDSSE open dataset | github.com/floodlight-sports (CC-BY 4.0) | DFL official per-frame `possession` and `ballstatus` codes (human-validated event feed) | 7 Bundesliga matches 2022/23, 25 Hz, ~2.0 M frames after home/away mirroring | training + leave-one-match-out CV of the HGB, BiGRU and PathCRF; choice of switch penalties |
| Metrica Sports sample data | github.com/metrica-sports/sample-data | team possession derived by us from Metrica's human-coded events | games 1+2, ~0.3 M frames | held-out sanity check only (never trained on) |
| PFF FC World Cup 2022 | PFF FC public release (free sign-up) | algorithmic labels written by us from PFF's human-coded on-ball events (rule in `src/pff.py`) | 64 matches, 25 fps resampled, ~10 M frames | PathCRF training only (DFL+PFF variant); **TODO(gpu)** kept or dropped per fold results. Also relabelled by a teacher model trained on DFL (`src/label_transfer.py`), tested for the frame models and dropped |
| PathCRF authors' checkpoint `saved/140` | github.com/hyunsungkim-ds/pathcrf (KDD 2026) | trained by the authors on the same DFL open dataset (leave-one-out, held-out J03WR9) | — | reference only; the submitted model is our own retraining (trial 400) |
| SoccerNet-Echoes + SoccerNet labels | github.com/SoccerNet | Whisper transcripts; SoccerNet human action labels | 330 matches | commentary stoppage model (dropped, see §5) |
| BAR–ATH tactical ATD | provided | automatic | — | inference only; also the source of algorithmic ball-detector tiles (dropped, §5) |

DFL ball visibility is ~100%, so during training the ball is hidden on a schedule matching
BAR–ATH's gap statistics (57% visible) and home/away are mirrored.

## 5. Assets tried and dropped (all documented in `eda/NOTES.md`)
| asset | what we did | why dropped |
|---|---|---|
| broadcast video + broadcast ATD | OCR'd the scoreboard clock to align broadcast to tactical frames (±0.5 s); tried fusing the broadcast ball | broadcast ball covers 26% of graded frames and adds 2.6% new frames |
| broadcast audio | referee-whistle detector on the Demucs crowd stem; restart words from local Whisper commentary transcripts | the crowd whistles constantly: P(dead | whistle) 0.14 vs base 0.20 |
| commentary stoppage model | TF-IDF + logistic regression trained on SoccerNet-Echoes vs SoccerNet stoppage labels (other games) | held-out AUC 0.57; too weak to add as a decode-time prior |
| own ball detector (YOLOv8n on tactical frames, labels projected from the ATD ball, no human labels) | fills ball gaps from 59% to ~78% visibility | neutral once judged honestly: a simulator of the detector's error profile on the DFL leave-one-out moves spellF1 by +0.006 at best, and reproduction would need a 2.5-h detector pass |
| smart tagging | SET PIECES windows as dead-ball evidence | only 56% of their frames are dead; no restart moment readable |
| PFF for the HGB / BiGRU | added 64 matches to training, first with a hand event-to-label rule, then with labels from a teacher model trained on DFL to reproduce the official labels (0.75 / 0.95 vs official) | neutral on held-out DFL (0.438 → 0.429 rule, 0.431 teacher): data volume is not the frame model's bottleneck |

## 6. Results (held-out external matches; public 5 min reported as sanity only)
| model | DFL leave-one-match-out spellF1 / macroF1 | public 5 min (sanity) |
|---|---|---|
| organisers' baseline_smooth50 | 0.12 / 0.44 | 0.33 / 0.62 |
| HGB v2 (38 features) | 0.50 / 0.89 | 0.52 / 0.70 |
| HGB v3 (46 features), frozen DFL dead ball | 0.56 / 0.91 | 0.39 / 0.69 |
| HGB v3 + dead-ball augmentation + 3-s minimum (DFL held-out also augmented) | 0.53 / 0.89 | 0.54 / 0.83 |
| BiGRU, frozen DFL dead ball | 0.60 / 0.91 | 0.47 / 0.67 |
| BiGRU + dead-ball augmentation + 3-s minimum | 0.58 / 0.91 | 0.64 / 0.76 |
| PathCRF authors' checkpoint + v2 dead balls | 0.69 / 0.95 (one match) | 0.62 / 0.72 |
| PathCRF authors' checkpoint + augmented HGB dead balls | – | 0.62 / 0.85 |
| **PathCRF ours, DFL only (7 folds)** | **TODO(gpu)** | TODO(gpu) |
| PathCRF ours, DFL + PFF (ablation fold) | TODO(gpu) | TODO(gpu) |
| **submitted: trial 400 + v2 dead balls** | — | TODO(gpu) |

## 7. Limitations
- Dead-ball onsets are still ~3 s late: the tracker's ball keeps moving after the whistle and the
  only cue is the players' behaviour. Restart ends are now within a frame or two.
- PathCRF expects 11 identified players per team; our stitched tracks have ~10.8 per frame and
  identity switches inside a track are unavoidable.
- Everything is calibrated on Bundesliga tempo; LaLiga differs.
- The 5-minute public window has 17 possessions, so all its numbers are within ±0.05 noise.
- With more time: a referee-whistle classifier trained on other games' audio, degrading DFL
  player tracking to BAR–ATH's slot/ID quality for honest model selection, and a PFF possession
  rule tuned against DFL so PFF can help the frame models too.

## 8. Reproduction
```bash
git clone <public repo>; cd <repo>; python3 -m venv .venv; .venv/bin/pip install -r requirements.txt
# unpack the data bundle into data/ as in README.md
.venv/bin/python src/v2.py --out evaluation/results/v2.csv     # dead balls (HGB, weights committed)
bash pod/pathcrf_setup.sh                                        # PathCRF code+patch+venv, our weights (GitHub release), inference
cp evaluation/results/pathcrf.csv submission/<team_name>.csv
python evaluation/validate_submission.py submission/<team_name>.csv
```
Retraining: `src/v2.py --train` (2 min, laptop), `src/seq.py --train` (25 min), PathCRF via
`pod/vast.sh` on one RTX 4090 (~85 GPU-hours for the 7 DFL folds, the PFF ablation fold and the
final model). Fold logs are committed under `pod/logs/`.
