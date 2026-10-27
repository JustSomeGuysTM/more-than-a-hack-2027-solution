# MTAH 2027 — team notes (Elijah, Oct 25)

## The task in one line
One LaLiga match (Barcelona vs Athletic, 2024-25 MD2). For each of the 199,319 frames of the
tactical video (25 fps) output `H` (Barça has ball), `A` (Athletic has ball), `D` (dead ball),
or `X` (abstain). Scored on the private full-match truth; we only see the first 5 min of play.

## Scoring — this drives everything
- Ranked on **spellF1**: each possession (maximal run of H or A) is one object. Predicted
  possessions are matched 1:1 to true ones by temporal IoU at thresholds 0.3..0.7, F1 averaged.
  Constant guess = 0. A possession 1 s late scores 0.65 on its own. Noise band ~0.04.
- **macroF1** (per-frame, mean of H/A/D F1) is the tie-break only.
- `X` is forward-filled with our previous label before spellF1 runs. So X inside a possession is
  free; X across a hand-over is as bad as being wrong.
- Baseline (nearest player to ball + 2 s majority vote) = spellF1 0.2433 full match. Beat that.
- Top 20 advances. Deadline Nov 17. Deliverables: CSV + public repo that regenerates it + short
  technical doc.

## Rules that bite
- No hand labels on this game, for anything (train, validate, fix). Pseudo-labels from an
  algorithm are fine. Hand-labelled OTHER games are fine.
- No event data / corrected tracking about this game from anywhere.
- **Do not upload the video or data to hosted LLM APIs** (terms of use). Local models only
  (whisper via mlx, local YOLO, etc). This also rules out Polyglot's Claude translate backend.
- Public submission repo must not contain data/ or derived frame-level exports.

## The data (all in `data/`, numbers verified locally)
| asset | what | notes |
|---|---|---|
| tactical ATD (`_atd.txt` + `.xml`) | 199,319 lines, `frame:p1;p2;..;p46:ball`, each `x,y,speed`, 0-1 normalised | row i = video frame i-1. 46 track *slots*, not players; slots 1-23 Barça, 24-46 Athletic (teamId in xml). Ball visible 56.9% all / 59.3% graded. ~20.5 players tracked per graded frame |
| tactical homography | 3x3 pitch(0-1)->pixel per frame, 100% valid | lets us project tracking onto the video |
| smart tagging xml | 638 phases (`BUILD UP`, `SET PIECES`...), seconds from video start, with `Team` tag | `Team` = N/A on all throw-ins. Phases, not possessions |
| broadcast ATD + homography | 164,488 frames, own timeline, 65.8% valid homography | not synced to tactical; has the commentary audio |
| videos | tactical 199,319 f; broadcast 164,488 f; ~13 GB | downloading into `data_raw/` |

Graded frames: 11185–90651 (1st half) and 110808–190197 (2nd half). 2nd half kicks off at
113742 — the 2,934 frames before that are graded and all `D`.

## What the public 5 minutes shows (`eda/out/timeline_public.png`)
- 21 segments / 17 possessions, mean 14 s, but possessions as short as **0.76 s and 1.3 s** exist
  (GT frames 15346-15364, 15766-15798). A 2 s majority vote erases those by construction.
- Nearest-player team agrees with GT on only 66% of alive frames with a ball: every pass flips it.
- Dead ball ≠ missing ball: in this window the ball is *visible* on 66% of D frames, and missing
  on 6.5% of alive frames. The baseline's "no ball → D" is the wrong proxy.
- Dead ball looks like: ball far from every player (nearest-player distance 10-15 m) and/or ball
  outside the pitch rectangle, with players walking. Smart tagging `SET PIECES` marks the restart.
- Smart tagging `Team` is roughly right at phase level but lags transitions by seconds and is
  blank during set-piece build-ups.

## Plan (proposal, individual week then converge)
1. **Feature table per frame** (numpy, cached in `eda/atd_tactical.npz` already): ball xy/speed,
   nearest player per team + distance, ball in/out of pitch, ball visible, smart-tag phase/team,
   n tracked players, team centroids / spread, ball velocity direction vs team.
2. **Dead-ball detector first** (D vs alive). It's the biggest easy win: ball off pitch, players
   static, SET PIECES tag, 2nd-half pre-kickoff. Whistle detection from audio if tactical video
   has audio (check `ffprobe` when the mp4 lands).
3. **Possession as a segmentation problem, not a per-frame classifier.** Detect *touch events*
   (ball speed/direction change near a player), assign each touch a team, then a possession
   changes when the opposing team gets N consecutive touches / holds for T seconds. Tune N,T on
   the whole-match *structure* (possession count, duration distribution look like football), not
   on the 5 minutes. HMM / Viterbi over {H,A,D} with touch-based emissions is the clean version.
4. **Ball gap filling**: interpolate short gaps with constant velocity; for long gaps, the player
   cluster motion tells you where play is.
5. Only then: broadcast video + commentary (Polyglot's mlx-whisper + Demucs path, local), needs
   a broadcast→tactical time alignment (match on player-cluster shape through both homographies).

## Commentary (Polyglot, local only)
Broadcast mp4 audio -> ffmpeg 16 kHz wav -> Demucs two-stems (vocals = commentators, no_vocals =
crowd + whistle) -> mlx-whisper large-v3-turbo with word timestamps -> `eda/audio/commentary_*.jsonl`.
Times are broadcast seconds; mapping to tactical frames needs the broadcast->tactical alignment.
Only Polyglot's mlx-whisper + Demucs path is allowed; its Gemini-audio and Claude backends upload
the audio and are off limits for this data.

**Done (2026-10-25):** `~/Polyglot/.venv/bin/python eda/commentary.py --source broadcast` ran on
the full match in ~40 min (Demucs ~30 min, Whisper ~8 min). English commentary, 5,020 segments /
18,010 words with word timestamps -> `eda/audio/commentary_broadcast.jsonl`; stems in
`eda/audio/stems/htdemucs/broadcast/{vocals,no_vocals}.wav` (1.2 GB each, gitignored; `no_vocals`
is the crowd + whistle track for a whistle detector). Mostly colour commentary: play-by-play words
are sparse (corner 21, offside 13, foul 19, goal 88, throw 4, free kick 1 mentions) so it will be a
restart/stoppage cue, not a frame-level possession signal. Timestamps map to tactical frames with
`eda.align.b2t` (see alignment section).

## databallpy
`eda/dbp.py` builds a real databallpy `Game` from the numpy cache (its own loaders reject this
export). Everything on https://databallpy.readthedocs.io/en/latest/getting_started/visualizations_page.html
then works: `plot_tracking_data`, `save_tracking_video`, `add_velocity`, `get_pitch_control`.
Placeholders to replace with our model's output: `ball_status` (= ball tracked) and
`player_possession` (= nearest player). Team colours default to green=home, red=away; pass
`team_colors=["#d62728", "#1f77b4"]` to match `eda/render.py`.
```python
from eda.dbp import game
g = game(); td = g.tracking_data
td.add_velocity(g.get_column_ids() + ["ball"], filter_type="savitzky_golay", max_velocity=50)
from databallpy.visualize import plot_tracking_data, save_tracking_video
plot_tracking_data(g, 11260, add_velocities=True, add_player_possession=True)
save_tracking_video(g, 11185, 11685, "eda/out", title="clip", add_velocities=True, add_player_possession=True)
```

## Videos
`data/tactical/BAR-ATH_tactical.mp4` (5.2 GB, from the team bucket `s3://more-than-a-hack-2027`,
profile `mtah`). 199,319 frames at 25 fps, **has a stereo AAC audio track** -> whistle / crowd
detection on the submission timeline with no alignment problem. **Checked: the track is near-silent**
(mean -58 dB, peak -44 dB; whisper hallucinates "Thank you" on it) -> no commentary, probably no
usable whistle either. Commentary comes from `data/broadcast/BAR-ATH_broadcast.mp4` (13 GB, same
bucket, `broadcast/` prefix).

## Running things
```
.venv/bin/python eda/atd.py                 # builds/validates the numpy cache (5 s)
.venv/bin/python eda/plot.py 11185 15889    # pitch PNGs -> eda/out/
.venv/bin/python eda/plot.py --clip 11185 11600   # GIF
.venv/bin/python eda/timeline.py            # the 5-min comparison figure
.venv/bin/python eda/render.py --source tactical  --start 11185 --end 11685          # pitch-view mp4
.venv/bin/python eda/render.py --source tactical --start 11185 --end 11685 --video --possession --trail   # camera + overlay | pitch
.venv/bin/python eda/render.py --source tactical --start 11185 --end 18685 --velocities --possession --trail --heatmap   # all overlays
.venv/bin/python eda/dbp.py                 # databallpy smoke test -> eda/out/dbp_*
~/Polyglot/.venv/bin/python eda/commentary.py --start 600 --end 900                  # local whisper on broadcast audio
.venv/bin/python starter/baseline.py        # regenerate the 4 baselines
.venv/bin/python evaluation/run_evaluation.py   # score everything in evaluation/results/
```

## External data (allowed: RULES.md "External data about other games", FAQ "must be about other games")
Hard line: **nothing human-made about BAR-ATH itself** — so ESPN/FotMob/Opta play-by-play *of this
match* is out. Anything from other matches is fine as long as it is public and documented in the
tech doc (source, who labelled, how much). Candidates, best first:

| source | what | why we care |
|---|---|---|
| DFL / IDSSE open dataset (CC-BY 4.0, 7 Bundesliga matches 2022/23, 25 Hz, ~2.4 GB) | tracking **+ per-frame `possession` (team) and `ballstatus` (alive/dead) codes** + events; `pip install floodlight`, `IDSSEDataset().get(match_id)` | exactly our label space on real tracking -> train/validate the tracking->H/A/D model |
| Metrica sample-data (3 games, tracking+events synced, 105x68, same provider as ours) | events incl. passes/recoveries/ball lost per team | derive team possession per frame; same coordinate convention as our ATD |
| PFF FC 2022 World Cup (64 matches, broadcast tracking + events, free with signup) | possession events + broadcast-derived tracking | closest analogue to our *broadcast* ATD |
| SkillCorner open data (10 A-League matches, broadcast tracking, "Dynamic Events", phases of play) | no video, no events | broadcast-tracking gap behaviour |
| SoccerNet-Echoes (github SoccerNet/sn-echoes) | Whisper ASR transcripts of broadcast commentary per game, JSON {start,end,text}; SoccerNet video/labels need the SoccerNet NDA | commentary -> event/possession language model; SoccerNet Ball Action Spotting (12 classes incl. pass/drive/throw-in/out/free kick) gives event timestamps to align text to |
| GOAL (arXiv 2211.04534) | ~20 games, audio + human transcripts of live commentary | small, human-quality commentary |

Note: the tactical ATD smart-tagging + our own algorithmic pseudo-labels on BAR-ATH are also fine
(no human decided them).

## Broadcast -> tactical alignment (done, `eda/align.py`)
The LaLiga scoreboard clock is OCR'd (tesseract, 1 crop / 2 s, 80% of samples carry a clock;
the rest are replays/cutaways/pre-match). Clock runs in real time, so each half is one offset:

| half | clock = broadcast_sec + | tactical frame = | broadcast range with clock |
|---|---|---|---|
| 1 | -286.86 s | 11185 + 25·clock | 308–2978 s |
| 2 | -772.90 s | 113742 + 25·(clock-2700) | 3492–6544 s |

Refined to the frame (2026-10-26) by minimising the broadcast-ATD vs tactical-ATD ball
disagreement: clock-only offsets (-286.5 / -773.5) gave 2.4 / 4.0 m median disagreement, the
refined ones 0.82 / 0.86 m. Both values are kept in `eda/align.json` (`offset_clock`, `offset`).

Broadcast ball fusion was tried and dropped: the broadcast ATD carries a pitch-coordinate ball on
only 26% of graded frames and adds just 2.6% of frames the tactical feed does not already see.

91–94% of reads agree with the offset to within 1 s; the rest are OCR misreads. Validated
independently: broadcast-ATD vs tactical-ATD player-centroid distance is minimised within ±1–2 s of
the mapped frame. Uncertainty ≈ ±0.5 s (12 frames) from clock granularity. The clock display changes
format in stoppage time (reads stop at 44:50 / ~96:00) but the offset still applies there.
Use: `from eda.align import b2t; b2t(broadcast_seconds) -> tactical frame or None`. Constants in
`eda/align.json`; reads in `eda/out/ocr/clock_ocr.csv`. Tactical kickoffs confirmed from the ball
leaving the centre spot at frames 11185 and 113742.

## Possession v1 (`src/possession.py`, 2026-10-26)
Ball gap fill (≤2 s linear) -> per-frame evidence (H/A: `(8 m - dist to nearest player of that
team)`, D: off pitch / outside halves / ball missing / ball lying still away from everyone) ->
Viterbi over {H,A,D} with a switch penalty. numpy only, ~1 s for the whole match.
Tuned on the **7 DFL open matches** (real per-frame possession + ball-status labels) and **Metrica
sample games 1+2** (labels derived from their event stream), ball hidden to BAR-ATH-like 35%
visibility (`src/external.py`), never on the public window. DFL conversions cached in
`data_raw/external/dfl_*.npz` (12 s each to build with floodlight).

| | spellF1 | macroF1 | spells (gt) |
|---|---|---|---|
| 7 DFL matches degraded, organisers' baseline_smooth50 | 0.12 | 0.44 | 725 (437) |
| 7 DFL matches degraded, **v1** | **0.32** | 0.69 | 322 (437) |
| Metrica 1+2 degraded, baseline_smooth50 | 0.13 | 0.48 | 470 (346) |
| Metrica 1+2 degraded, **v1** | **0.44** | 0.67 | 309 (346) |
| BAR-ATH public 5 min, baseline_smooth50 (official scorer) | 0.33 | 0.62 | 26 (17) |
| BAR-ATH public 5 min, **v1** (sanity only) | **0.35** | 0.67 | 24 (17) |

Smart-tagging SET PIECES windows are weak D evidence only: 56% of their frames are dead in the
public GT, and the tracked ball is never "still" inside them (median 5 m/s, same as open play), so
the restart moment cannot be read off ball speed.

Still under-predicting spell count (misses short possessions) and D is rule-only. Next (v2):
train a per-frame classifier on DFL features (distances, ball speed, player spread, visibility) ->
P(H/A/D) as Viterbi emissions; touch-based switches; whistle (crowd stem) + commentary cues via
`eda.align.b2t` into the D evidence.

**S3:** my IAM user (`Eli-more-than-a-hack`) has no `s3:PutObject` on the bucket, so external data
is local only (`data_raw/external/`, gitignored) until Brandon grants write. Planned layout:
`s3://more-than-a-hack-2027/external/{metrica-sample,dfl}/`.

## Possession v2 (`src/v2.py`, `src/features.py`, 2026-10-26)
Per-frame HistGradientBoosting classifier (38 slot-agnostic features: per-team nearest distances
and counts, ball visibility/gap length, ball speed, pitch position, player spread/speed, ±0.5 s and
±2 s rolling context) trained on the **7 DFL matches** with the ball degraded, home/away mirrored
(2.0M rows). log P(H/A/D) are the Viterbi emissions, switch penalty 10. Model: `src/v2_model.pkl`
(3 MB, regenerate with `--train`, ~2 min). BAR-ATH inference: `python src/v2.py` (~30 s).

| | spellF1 | macroF1 | spells (gt) |
|---|---|---|---|
| DFL leave-one-match-out, v1 | 0.32 | 0.69 | 322 (437) |
| DFL leave-one-match-out, **v2** | **0.50** | **0.89** | 317 (437) |
| Metrica 1+2 degraded (never trained on), v1 | 0.44 | 0.67 | 309 (346) |
| Metrica 1+2 degraded (never trained on), **v2** | **0.48** | 0.70 | 342 (346) |
| BAR-ATH public 5 min, baseline_smooth50 | 0.33 | 0.62 | 26 (17) |
| BAR-ATH public 5 min, v1 | 0.35 | 0.67 | 24 (17) |
| BAR-ATH public 5 min, **v2** (sanity only) | **0.52** | **0.70** | 16 (17) |

Frame accuracy on held-out DFL is 0.90; the DFL->Metrica gap (0.89 -> 0.70 macroF1) is partly the
derived Metrica labels, partly domain shift. Next: whistle + commentary cues as extra features
(via `eda.align.b2t`), smart-tagging windows as a feature, bigger context windows, and a
transition model that knows a possession starts with a controlled touch.

## Audio cues: negative result (`src/audio_cues.py`, 2026-10-26)
Whistle detector on the Demucs crowd stem (3-4.5 kHz tonal peak, robust z > 6, 227 onsets) and
commentary restart words (throw-in / corner / free kick / foul / offside / penalty / goal, word
timestamps mapped with `eda.align.b2t`). Checked against the smart-tagging SET PIECES windows
(no labels involved): P(set-piece window | whistle) 0.26 vs base 0.22, restart words 0.27. In the
public window P(D | whistle) is 0.14 vs base 0.20 -- the stadium crowd whistles constantly, so the
detector finds fans, not the referee. Not used in v3. Would need a referee-whistle classifier
trained on other games' audio (e.g. SoccerNet-Echoes games) to be worth revisiting.

## Possession v3 (`src/v2.py` + 8 "since/until" features, 2026-10-26)
Adds seconds since/until the last home touch, away touch, ball off pitch and ball speed jump
(46 features). ±5 s rolling context was also tried: no gain, dropped.

| | spellF1 | macroF1 | spells (gt) |
|---|---|---|---|
| DFL leave-one-match-out, v2 -> **v3** | 0.50 -> **0.56** | 0.89 -> **0.91** | 317 -> 316 (437) |
| Metrica 1+2 degraded (held out), v2 -> **v3** | 0.48 -> **0.52** | 0.70 -> **0.71** | 342 -> 320 (346) |
| BAR-ATH public 5 min, v2 -> v3 (sanity only) | 0.52 -> 0.39 | 0.70 -> 0.69 | 16 -> 27 (17) |

Split result: every external check improves, the public window gets worse and v3 fragments
BAR-ATH into 27 spells. 17 possessions is far too few to decide, but the fragmentation is a real
hint that the touch features behave differently on our tracklets (46 slots, ~20.5 tracked, ID
switches) than on DFL's clean 22 tracks. **v2 stays the submission candidate** until the external
validation also degrades the *player* tracking to BAR-ATH-like quality (drop ~7% of player
observations, add ID churn) -- that is the honest way to pick between them.

## PathCRF (KDD 2026, github.com/hyunsungkim-ds/pathcrf) — first honest number (2026-10-26)
Cloned to `~/pathcrf` (own venv: torch, torch_geometric, rdp). Its training data *is* the DFL open
dataset (raw XML linked in from floodlight's cache, `datatools/preprocess.py` run). Authors' pretrained
checkpoint `saved/140` (leave-one-match-out, held-out = J03WR9) run on that match, 70 s on CPU.
It predicts the possession edge (carrier -> receiver) from **player tracking only, no ball**.

| J03WR9 held out, GT dead frames substituted for both | spellF1 | macroF1 | spells (gt 370) | team acc alive |
|---|---|---|---|---|
| GBDT v3 trained on the other 6 | 0.57 | 0.93 | 278 | 0.90 |
| **PathCRF** pretrained | **0.69** | **0.95** | **366** | **0.93** |

PathCRF gets the possession *structure* right (366 vs 370 spells) where our model merges short
possessions. It does not model dead balls -> pair it with our D detector. Open problem: it wants
11 identified players per team; BAR-ATH has 46 tracklet slots with ID churn (see src/pathcrf_io.py
once written). One-line patch applied in `~/pathcrf/inference.py` (pandas read-only arrays).
