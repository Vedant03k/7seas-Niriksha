from audio_detector import AudioDeepfakeDetector
import os

det = AudioDeepfakeDetector()

dl = r"C:\Users\VEDANT\Downloads\Niki"

# Test ElevenLabs v2 files
for f in ["ankit2.mp3", "Ankitfake voice.mp3", "BankScamaudio.mp3", "Eleindian.mp3"]:
    path = os.path.join(dl, f)
    if os.path.exists(path):
        r = det.analyze_audio(path)
        v = r.get("verdict", "ERR")
        c = r.get("confidence", 0)
        ss = r.get("sub_scores", {})
        print(f"=== {f}: {v} ({round(c*100,1)}%) ===")
        for k, val in ss.items():
            print(f"  {k}: {round(val*100,1)}%")
        for a in r.get("artifacts_detected", [])[:5]:
            print(f"  -> {a}")
        print()

# Also test real recordings
print("--- REAL RECORDINGS ---")
for f in ["RealRecording.m4a", "RealRecording3.m4a", "Recording.mp3"]:
    path = os.path.join(dl, f)
    if os.path.exists(path):
        r = det.analyze_audio(path)
        ss = r.get("sub_scores", {})
        print(f"{f}: {r.get('verdict','ERR')} ({round(r.get('confidence',0)*100,1)}%)")
        for k, val in ss.items():
            print(f"  {k}: {round(val*100,1)}%")
        for a in r.get("artifacts_detected", [])[:3]:
            print(f"  -> {a}")
        print()

