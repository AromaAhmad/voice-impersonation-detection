from load_protocols import load_split, FILES
import pandas as pd

df = pd.concat([load_split(s) for s in FILES], ignore_index=True)

print("Attack ke hisaab se ginti (split x attack):")
spoof = df[df.label == "spoof"]
print(spoof.groupby(["split", "attack"]).size().unstack(0).fillna(0).astype(int))

tr = set(df[df.split == "train"].speaker)
dv = set(df[df.split == "dev"].speaker)
ev = set(df[df.split == "eval"].speaker)
print("\nSpeaker overlap train&dev :", len(tr & dv))
print("Speaker overlap train&eval:", len(tr & ev))
print("Speaker overlap dev&eval  :", len(dv & ev))
