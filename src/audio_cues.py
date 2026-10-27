"""BAR-ATH-only cues from the broadcast audio, on the tactical frame timeline (via eda.align.b2t).

    .venv/bin/python src/audio_cues.py      # -> eda/out/cues_tactical.npz  {whistle (F,), restart_word (F,)}

whistle: referee whistle = narrow tonal peak around 3-4.5 kHz in the Demucs crowd stem. Per 1/25 s
frame: band-to-total energy ratio x spectral peakiness, robust z-scored; an onset is >= 5 frames above
Z_ON. Unsupervised -- no labels of this game are involved; checked against smart-tagging restarts.
restart_word: commentary words that name a stoppage (throw-in, corner, free kick, goal kick, offside,
foul, penalty, handball, goal) from eda/audio/commentary_broadcast.jsonl, word-timestamped.
"""
import json, os, subprocess, sys
import numpy as np
from scipy.signal import stft
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from eda.align import b2t
STEM = os.path.join(ROOT, "eda/audio/stems/htdemucs/broadcast/no_vocals.wav"); JSONL = os.path.join(ROOT, "eda/audio/commentary_broadcast.jsonl")
OUT = os.path.join(ROOT, "eda/out/cues_tactical.npz"); F_TACT = 199319; SR = 16000; HOP = SR // 25
WORDS = ("throw", "corner", "free kick", "free-kick", "goal kick", "offside", "foul", "penalty", "handball", "goal", "half-time", "half time")
Z_ON = 6.0

def whistle_score():
    pcm = subprocess.run(["ffmpeg", "-v", "error", "-i", STEM, "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"], capture_output=True).stdout
    x = np.frombuffer(pcm, np.float32); scores = []
    for s in range(0, len(x), SR * 600):                                   # 10-minute chunks
        f, t, Z = stft(x[s:s + SR * 600 + 1024], SR, nperseg=1024, noverlap=1024 - HOP, boundary=None); P = np.abs(Z) ** 2
        band = (f >= 2800) & (f <= 4600); pb = P[band]
        ratio = pb.sum(0) / (P.sum(0) + 1e-9); peak = pb.max(0) / (pb.mean(0) + 1e-12)
        scores.append(ratio * peak)
    sc = np.concatenate(scores); med = np.median(sc); mad = np.median(np.abs(sc - med)) + 1e-9
    return (sc - med) / (1.4826 * mad)                                   # robust z, one value per 1/25 s of broadcast

def onsets(z, n=5):
    on = z > Z_ON; run = np.convolve(on, np.ones(n), "same") >= n; return np.flatnonzero(run[1:] & ~run[:-1]) + 1

def main():
    z = whistle_score(); ons = onsets(z); print("whistle onsets:", len(ons))
    whistle = np.zeros(F_TACT, np.float32); restart = np.zeros(F_TACT, np.float32)
    for i in ons:
        f = b2t(i / 25)
        if f is not None: whistle[max(f - 12, 0):f + 50] = 1.0                 # -0.5 s .. +2 s around the onset
    for line in open(JSONL):
        s = json.loads(line)
        for w in s["words"]:
            if any(k in w["word"].lower() for k in WORDS) or any(k in s["text"].lower() and w is s["words"][0] for k in WORDS if " " in k):
                f = b2t(w["start"])
                if f is not None: restart[max(f - 100, 0):f + 50] = 1.0        # commentary lags play: -4 s .. +2 s
    np.savez_compressed(OUT, whistle=whistle, restart_word=restart, whistle_z=z.astype(np.float32))
    import xml.etree.ElementTree as ET; from games import require
    sp = np.zeros(F_TACT, bool)
    for i in ET.parse(require(["smart_tagging"])[0]).getroot().iter("instance"):
        if i.findtext("code") == "SET PIECES": sp[int(float(i.findtext("start")) * 25):int(float(i.findtext("end")) * 25) + 1] = True
    print("frames flagged: whistle %.3f restart_word %.3f setpiece %.3f" % (whistle.mean(), restart.mean(), sp.mean()))
    print("P(setpiece window | whistle flag) %.2f  P(setpiece | restart_word) %.2f  base P(setpiece) %.2f" % (sp[whistle > 0].mean(), sp[restart > 0].mean(), sp.mean()))
    print("wrote", OUT)

if __name__ == "__main__":
    main()
