import pandas as pd
from pathlib import Path
from load_protocols import load_split

SEED = 42   # fixed seed: dobara chalane par wahi subset aaye
out = Path("data/subsets"); out.mkdir(parents=True, exist_ok=True)

def sample_balanced(df, n_real, n_spoof):
    real = df[df.label == "bonafide"].sample(n_real, random_state=SEED)
    spoof = df[df.label == "spoof"]
    per_attack = n_spoof // spoof.attack.nunique()
    spoof = (spoof.groupby("attack", group_keys=False)
                  .apply(lambda g: g.sample(per_attack, random_state=SEED)))
    return pd.concat([real, spoof]).sample(frac=1, random_state=SEED)

train = sample_balanced(load_split("train"), 2000, 1998)   # 333 x 6
dev   = sample_balanced(load_split("dev"),    500,  498)   # 83 x 6
ev    = sample_balanced(load_split("eval"),  1000, 1950)   # 150 x 13

for name, d in [("train", train), ("dev", dev), ("eval", ev)]:
    d.to_csv(out / f"{name}.csv", index=False)
    print(name, len(d), d.label.value_counts().to_dict())
print("Saved in", out)