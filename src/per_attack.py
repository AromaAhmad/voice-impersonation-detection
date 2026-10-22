import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.metrics import roc_curve
from load_protocols import load_split

def load(split):
    df = pd.read_csv(f"data/features/{split}.csv", keep_default_na=False)
    X = df.iloc[:, 4:].values.astype(float)
    y = (df["label"] == "spoof").astype(int).values
    return df, X, y

def eer(y, score):
    fpr, tpr, _ = roc_curve(y, score)
    fnr = 1 - tpr
    i = np.argmin(np.abs(fnr - fpr))
    return (fpr[i] + fnr[i]) / 2

_, Xtr, ytr = load("train")
ev_df, Xev, yev = load("eval")

model = make_pipeline(StandardScaler(), SVC(kernel="rbf", class_weight="balanced"))
model.fit(Xtr, ytr)
score = model.decision_function(Xev)

proto = load_split("eval").set_index("file")["attack"]
attack = ev_df["file"].map(proto).astype(str).values

real = yev == 0
print("Total eval:", len(yev), "| real:", real.sum(), "| fake:", (~real).sum())
print("Attacks:", sorted(set(attack[~real])))

rows = []
for a in sorted(set(attack[~real])):
    fake_a = (attack == a) & (~real)
    n = int(fake_a.sum())
    if n == 0:
        print("SKIP", a)
        continue
    mask = real | fake_a
    rows.append({"attack": a, "n_fake": n, "EER": round(float(eer(yev[mask], score[mask])), 3)})

out = pd.DataFrame(rows)
print(out.to_string(index=False))
out.to_csv("results/per_attack_eer_svm.csv", index=False)
print("Overall EER:", round(float(eer(yev, score)), 3))
