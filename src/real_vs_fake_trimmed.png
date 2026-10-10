import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

files = [("REAL (meri awaaz)", "data/my_voice/clip1.wav"),
         ("FAKE (TTS)", "data/my_voice/fake1.wav")]

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
for ax, (title, path) in zip(axes, files):
    y, sr = librosa.load(path, sr=16000)   # dono ko 16 kHz par laate hain
    y, _ = librosa.effects.trim(y, top_db=40)   # shuru aur aakhir ki khamoshi hatao
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
    S_db = librosa.power_to_db(S, ref=np.max)
    img = librosa.display.specshow(S_db, sr=sr, x_axis="time", y_axis="mel", ax=ax)
    ax.set_title(f"{title} - {len(y)/sr:.1f} sec")
    fig.colorbar(img, ax=ax, format="%+2.0f dB")

plt.tight_layout()
Path("results").mkdir(exist_ok=True)
plt.savefig("results/real_vs_fake_mine.png", dpi=150)
plt.show()