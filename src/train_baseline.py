import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_curve)

SEED = 42
Path("results").mkdir(exist_ok=True)

def load(split):
    df = pd.read_csv(f"data/features/{split}.csv")
    X = df.iloc[:, 4:].values            # pehle 4 columns metadata hain
    y = (df["label"] == "spoof").astype(int).values   # fake = 1
    return df, X, y

def eer(y, score):
    fpr, tpr, _ = roc_curve(y, score)
    fnr = 1 - tpr
    i = np.nanargmin(np.abs(fnr - fpr))
    return (fpr[i] + fnr[i]) / 2

tr_df, Xtr, ytr = load("train")
dv_df, Xdv, ydv = load("dev")
ev_df, Xev, yev = load("eval")

models = {
    "SVM": make_pipeline(StandardScaler(), SVC(kernel="rbf", class_weight="balanced")),
    "RandomForest": RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                           random_state=SEED, n_jobs=4),
}

def get_score(model, X):
    return model.decision_function(X) if hasattr(model, "decision_function") \
        else model.predict_proba(X)[:, 1]

rows = []
fitted = {}
for name, m in models.items():
    m.fit(Xtr, ytr)
    fitted[name] = m
    for split, X, y in [("dev", Xdv, ydv), ("eval", Xev, yev)]:
        pred = m.predict(X)
        sc = get_score(m, X)
        tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
        rows.append(dict(model=name, split=split,
                         accuracy=accuracy_score(y, pred),
                         precision=precision_score(y, pred),
                         recall=recall_score(y, pred),
                         f1=f1_score(y, pred),
                         EER=eer(y, sc),
                         real_as_real=tn, real_as_fake=fp,
                         fake_as_real=fn, fake_as_fake=tp))

res = pd.DataFrame(rows).round(3)
print(res.to_string(index=False))
res.to_csv("results/baseline_results.csv", index=False)

# eval par har attack ka alag EER (sab real + us attack ke fake)
print("\nEval par har attack ka EER:")
real_mask = (ev_df["label"] == "bonafide").values
per = {}
for name, m in fitted.items():
    sc = get_score(m, Xev)
    out = {}
    for a in sorted(ev_df.loc[~real_mask, "attack"].unique()):
        mask = real_mask | (ev_df["attack"] == a).values
        out[a] = eer(yev[mask], sc[mask])
    per[name] = out
per_df = pd.DataFrame(per).round(3)
print(per_df.to_string())
per_df.to_csv("results/per_attack_eer.csv")