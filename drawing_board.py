"""
Hand Gesture-Based Virtual Drawing Board
Uses MediaPipe Tasks API (0.10.30+)
Controls:
  - Index finger up only   → Drawing mode
  - Index + Middle up      → Move mode / Select color
  - Thumb up only          → Clear canvas
  - Press 'q'              → Quit
  - Press 's'              → Save drawing as PNG
"""
import cv2
import numpy as np
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode


# ── Constants ────────────────────────────────────────────────────────────────
COLORS = {
    "Red":    (0, 0, 255),
    "Green":  (0, 255, 0),
    "Blue":   (255, 0, 0),
    "Yellow": (0, 255, 255),
    "White":  (255, 255, 255),
    "Eraser": (0, 0, 0),
}
COLOR_LIST  = list(COLORS.items())
PALETTE_X   = 10
PALETTE_Y   = 10
SWATCH_W    = 60
SWATCH_H    = 40
BRUSH_SIZE  = 8
MODEL_PATH  = "hand_landmarker.task"  # download with download_model.py

# ── Hand connections for drawing skeleton ────────────────────────────────────
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17)
]

# ── Finger state helper ───────────────────────────────────────────────────────
def fingers_up(landmarks, handedness_label, W, H):
    lm = landmarks
    def x(i): return lm[i].x
    def y(i): return lm[i].y

    fingers = []
    # Thumb
    if handedness_label == "Right":
        fingers.append(x(4) < x(3))
    else:
        fingers.append(x(4) > x(3))
    # Index, Middle, Ring, Pinky
    for tip, pip in [(8,6),(12,10),(16,14),(20,18)]:
        fingers.append(y(tip) < y(pip))
    return fingers  # [thumb, index, middle, ring, pinky]

# ── Draw skeleton manually ───────────────────────────────────────────────────
def draw_landmarks(frame, landmarks, W, H):
    pts = [(int(lm.x * W), int(lm.y * H)) for lm in landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (0, 255, 0), 2)
    for pt in pts:
        cv2.circle(frame, pt, 4, (0, 0, 255), -1)

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    canvas     = np.zeros((H, W, 3), dtype=np.uint8)
    draw_color = COLORS["Red"]
    prev_point = None
    mode_text  = ""
    mode_time  = 0

    # ── Load HandLandmarker model ─────────────────────────────────────────
    options = HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_tracking_confidence=0.6
    )

    with HandLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Convert to MediaPipe Image
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )

            detection = landmarker.detect(mp_image)

            # ── Draw color palette ────────────────────────────────────────
            for i, (name, color) in enumerate(COLOR_LIST):
                x1 = PALETTE_X + i * (SWATCH_W + 5)
                y1 = PALETTE_Y
                x2 = x1 + SWATCH_W
                y2 = y1 + SWATCH_H
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (200, 200, 200), 2)
                if color == draw_color:
                    cv2.rectangle(frame, (x1-3, y1-3), (x2+3, y2+3), (255,255,255), 3)
                cv2.putText(frame, name, (x1+2, y2+15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200,200,200), 1)

            # ── Process detections ────────────────────────────────────────
            if detection.hand_landmarks:
                for i, landmarks in enumerate(detection.hand_landmarks):
                    # Get handedness
                    handedness_label = detection.handedness[i][0].display_name

                    draw_landmarks(frame, landmarks, W, H)

                    fx = int(landmarks[8].x * W)
                    fy = int(landmarks[8].y * H)

                    f = fingers_up(landmarks, handedness_label, W, H)

                    # CLEAR: only thumb up
                    if f[0] and not any(f[1:]):
                        canvas     = np.zeros((H, W, 3), dtype=np.uint8)
                        prev_point = None
                        mode_text  = "CLEARED!"
                        mode_time  = time.time()

                    # MOVE: index + middle up
                    elif f[1] and f[2] and not f[3] and not f[4]:
                        prev_point = None
                        mode_text  = "MOVE"
                        mode_time  = time.time()
                        cv2.circle(frame, (fx, fy), BRUSH_SIZE+4, draw_color, 2)

                        # Color selection from palette
                        if fy < PALETTE_Y + SWATCH_H + 20:
                            for j, (name, color) in enumerate(COLOR_LIST):
                                x1 = PALETTE_X + j * (SWATCH_W + 5)
                                if x1 <= fx <= x1 + SWATCH_W:
                                    draw_color = color

                    # DRAW: only index up
                    elif f[1] and not f[2] and not f[3] and not f[4]:
                        mode_text = "DRAW"
                        mode_time = time.time()
                        cv2.circle(frame, (fx, fy), BRUSH_SIZE, draw_color, -1)
                        if prev_point:
                            cv2.line(canvas, prev_point, (fx, fy), draw_color, BRUSH_SIZE * 2)
                        prev_point = (fx, fy)

                    else:
                        prev_point = None
            else:
                prev_point = None

            # ── Blend canvas onto frame ───────────────────────────────────
            gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
            _, mask    = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
            mask_inv   = cv2.bitwise_not(mask)
            frame_bg   = cv2.bitwise_and(frame, frame, mask=mask_inv)
            canvas_fg  = cv2.bitwise_and(canvas, canvas, mask=mask)
            frame      = cv2.add(frame_bg, canvas_fg)

            # ── Mode label ────────────────────────────────────────────────
            if time.time() - mode_time < 1.5:
                cv2.putText(frame, mode_text, (W//2 - 60, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,255), 3)

            cv2.putText(frame, "q=quit  s=save", (W-200, H-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150,150,150), 1)

            cv2.imshow("Virtual Drawing Board", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                fname = f"drawing_{int(time.time())}.png"
                cv2.imwrite(fname, canvas)
                print(f"Saved: {fname}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()