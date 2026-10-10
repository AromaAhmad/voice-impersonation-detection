import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

y, sr = librosa.load("data/my_voice/clip1.wav", sr=16000)

S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
S_db = librosa.power_to_db(S, ref=np.max)

fig, axes = plt.subplots(2, 1, figsize=(10, 7))

librosa.display.waveshow(y, sr=sr, ax=axes[0])
axes[0].set_title("Waveform (waqt vs awaaz ki tezi)")

img = librosa.display.specshow(S_db, sr=sr, x_axis="time", y_axis="mel", ax=axes[1])
axes[1].set_title("Mel spectrogram (waqt vs frequency)")
fig.colorbar(img, ax=axes[1], format="%+2.0f dB")

plt.tight_layout()
Path("results").mkdir(exist_ok=True)
plt.savefig("results/my_voice_clip1.png", dpi=150)
plt.show()