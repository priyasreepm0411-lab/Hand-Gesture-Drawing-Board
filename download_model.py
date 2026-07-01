"""
Run this once to download the MediaPipe hand landmarker model.
  python download_model.py
"""
import urllib.request, os

URL  = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
DEST = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")

if os.path.exists(DEST):
    print(f"Model already exists: {DEST}")
else:
    print("Downloading hand_landmarker.task ...")
    urllib.request.urlretrieve(URL, DEST)
    print(f"Saved to: {DEST}")