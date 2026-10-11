import pandas as pd
from pathlib import Path

ROOT = Path("data/ASVspoof2019/LA")
PROTO = ROOT / "ASVspoof2019_LA_cm_protocols"

FILES = {
    "train": "ASVspoof2019.LA.cm.train.trn.txt",
    "dev":   "ASVspoof2019.LA.cm.dev.trl.txt",
    "eval":  "ASVspoof2019.LA.cm.eval.trl.txt",
}

def load_split(split):
    df = pd.read_csv(
        PROTO / FILES[split],
        sep=" ",
        header=None,
        names=["speaker", "file", "col3", "attack", "label"],
    )
    df["split"] = split
    df["path"] = df["file"].apply(
        lambda f: str(ROOT / f"ASVspoof2019_LA_{split}" / "flac" / f"{f}.flac")
    )
    return df

if __name__ == "__main__":
    all_df = pd.concat([load_split(s) for s in FILES], ignore_index=True)

    print("Real/Fake ginti har partition mein:")
    print(all_df.groupby(["split", "label"]).size().unstack())

    print("\nSpeakers har partition mein:")
    print(all_df.groupby("split")["speaker"].nunique())

    print("\nAttack types har partition mein:")
    print(all_df[all_df.label == "spoof"].groupby("split")["attack"].unique())

    # pehli file maujood hai ya nahi, ye bhi check
    first = Path(all_df.iloc[0]["path"])
    print("\nPehli file maujood hai?", first.exists(), "->", first)