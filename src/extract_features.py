import numpy as np
import pandas as pd
import librosa
from joblib import Parallel, delayed
from pathlib import Path

SR = 16000

def extract(path):
    try:
        y, _ = librosa.load(path, sr=SR)
        mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=20)
        parts = [
            mfcc,
            librosa.feature.delta(mfcc),
            librosa.feature.spectral_centroid(y=y, sr=SR),
            librosa.feature.spectral_bandwidth(y=y, sr=SR),
            librosa.feature.spectral_rolloff(y=y, sr=SR),
            librosa.feature.spectral_flatness(y=y),
            librosa.feature.zero_crossing_rate(y),
            librosa.feature.rms(y=y),
        ]
        vec = []
        for m in parts:
            vec.extend(m.mean(axis=1))
            vec.extend(m.std(axis=1))
        return np.array(vec, dtype=np.float32)
    except Exception as e:
        print("ERROR", path, e)
        return None

def names():
    base = ([f"mfcc{i}" for i in range(20)] + [f"dmfcc{i}" for i in range(20)] +
            ["centroid", "bandwidth", "rolloff", "flatness", "zcr", "rms"])
    return [f"{b}_{s}" for b in base for s in ("mean", "std")]

if __name__ == "__main__":
    out = Path("data/features"); out.mkdir(parents=True, exist_ok=True)
    cols = names()
    for split in ["train", "dev", "eval"]:
        df = pd.read_csv(f"data/subsets/{split}.csv")
        print(f"{split}: {len(df)} files...")
        res = Parallel(n_jobs=4, verbose=5)(delayed(extract)(p) for p in df["path"])
        ok = [i for i, r in enumerate(res) if r is not None]
        X = pd.DataFrame(np.vstack([res[i] for i in ok]), columns=cols)
        meta = df.iloc[ok][["file", "speaker", "attack", "label"]].reset_index(drop=True)
        pd.concat([meta, X], axis=1).to_csv(out / f"{split}.csv", index=False)
        print(split, "saved:", X.shape)