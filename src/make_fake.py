import pyttsx3
from pathlib import Path

TEXT = "My name is Aroma and I am building a voice detection project."

out = Path("data/my_voice")
out.mkdir(parents=True, exist_ok=True)

engine = pyttsx3.init()
engine.save_to_file(TEXT, str(out / "fake1.wav"))
engine.runAndWait()
print("Saved:", out / "fake1.wav")
