import sounddevice as sd
import soundfile as sf
from pathlib import Path

SR = 16000   # sample rate: 1 second mein 16000 awaaz ke nuqte
DUR = 4      # recording ka waqt (seconds)

out = Path("data/my_voice")
out.mkdir(parents=True, exist_ok=True)

input("Enter dabao, phir 4 second tak kuch bolo...")
audio = sd.rec(int(DUR * SR), samplerate=SR, channels=1)
sd.wait()
sf.write(out / "clip1.wav", audio, SR)
print("Saved:", out / "clip1.wav")