import cv2
import numpy as np
import tensorflow as tf
import os
import sys
import time

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
IMG_SIZE       = 64          # must match training
MODEL_PATH     = "fight_model.h5"
VIDEO_FOLDER   = "test"
WINDOW_NAME    = "Fight Detection"
RESULT_HOLD_S  = 3           # seconds to show final result screen
FIGHT_THRESH   = 0.55        # if fight% > this → "Fight Detected"
SKIP_FRAMES    = 2           # predict every Nth frame (speed up)
DISPLAY_W      = 960         # window width
DISPLAY_H      = 540         # window height

# colours (BGR)
RED    = (0,   50,  220)
GREEN  = (0,  200,   50)
WHITE  = (255, 255, 255)
BLACK  = (0,     0,   0)
YELLOW = (0,   220, 220)
GRAY   = (80,   80,  80)

# ─────────────────────────────────────────────
#  LOAD MODEL
# ─────────────────────────────────────────────
if not os.path.exists(MODEL_PATH):
    print(f"[ERROR] Model not found: {MODEL_PATH}")
    sys.exit(1)

print("[INFO] Loading model …")
model = tf.keras.models.load_model(MODEL_PATH)
print("[INFO] Model loaded ✓")

classes = ["Fight", "NonFight"]

# ─────────────────────────────────────────────
#  HELPER: Draw HUD on frame
# ─────────────────────────────────────────────
def draw_hud(frame, label, confidence, fight_ratio, frame_no, total_frames, paused):
    h, w = frame.shape[:2]

    # ── top bar ──────────────────────────────
    cv2.rectangle(frame, (0, 0), (w, 70), (20, 20, 20), -1)

    color = RED if label == "Fight" else GREEN
    cv2.putText(frame, label, (20, 50),
                cv2.FONT_HERSHEY_DUPLEX, 1.6, color, 2)

    conf_text = f"Confidence: {confidence*100:.1f}%"
    cv2.putText(frame, conf_text, (w - 320, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, WHITE, 2)

    # ── bottom bar ────────────────────────────
    cv2.rectangle(frame, (0, h - 80), (w, h), (20, 20, 20), -1)

    # progress bar
    bar_x, bar_y, bar_w, bar_h = 20, h - 60, w - 40, 18
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), GRAY, -1)
    progress = int(bar_w * frame_no / max(total_frames, 1))
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + progress, bar_y + bar_h), (180, 130, 40), -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), WHITE, 1)

    # fight ratio bar (small, right side)
    ratio_x = w - 200
    cv2.putText(frame, "Fight%", (ratio_x, h - 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1)
    rb_x, rb_y, rb_w, rb_h = ratio_x + 65, h - 33, 115, 14
    cv2.rectangle(frame, (rb_x, rb_y), (rb_x + rb_w, rb_y + rb_h), GRAY, -1)
    filled = int(rb_w * fight_ratio)
    ratio_color = RED if fight_ratio > FIGHT_THRESH else GREEN
    cv2.rectangle(frame, (rb_x, rb_y), (rb_x + filled, rb_y + rb_h), ratio_color, -1)
    cv2.rectangle(frame, (rb_x, rb_y), (rb_x + rb_w, rb_y + rb_h), WHITE, 1)

    # frame counter
    cv2.putText(frame, f"Frame {frame_no}/{total_frames}", (20, h - 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1)

    # controls hint
    hint = "  [P] Pause   [N] Next   [Q] Quit"
    cv2.putText(frame, hint, (180, h - 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (160, 160, 160), 1)

    # paused overlay
    if paused:
        cv2.putText(frame, "⏸ PAUSED", (w // 2 - 100, h // 2),
                    cv2.FONT_HERSHEY_DUPLEX, 1.4, YELLOW, 3)

    return frame


# ─────────────────────────────────────────────
#  HELPER: Show final result screen
# ─────────────────────────────────────────────
def show_result_screen(video_name, final_label, fight_frames, total_frames):
    canvas = np.zeros((DISPLAY_H, DISPLAY_W, 3), dtype=np.uint8)
    canvas[:] = (20, 20, 20)

    color = RED if final_label == "Fight Detected" else GREEN
    icon  = "⚠  FIGHT DETECTED" if final_label == "Fight Detected" else "✓  NO FIGHT"

    # big result text
    cv2.putText(canvas, icon, (DISPLAY_W // 2 - 280, DISPLAY_H // 2 - 40),
                cv2.FONT_HERSHEY_DUPLEX, 1.8, color, 3)

    # stats
    pct = fight_frames / max(total_frames, 1) * 100
    stats = f"Fight frames: {fight_frames} / {total_frames}  ({pct:.1f}%)"
    cv2.putText(canvas, stats, (DISPLAY_W // 2 - 230, DISPLAY_H // 2 + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, WHITE, 2)

    # video name
    cv2.putText(canvas, f"Video: {video_name}", (DISPLAY_W // 2 - 230, DISPLAY_H // 2 + 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.70, (160, 160, 160), 1)

    cv2.putText(canvas, "Press [N] for next  |  [Q] to quit",
                (DISPLAY_W // 2 - 220, DISPLAY_H - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (120, 120, 120), 1)

    deadline = time.time() + RESULT_HOLD_S
    while time.time() < deadline:
        cv2.imshow(WINDOW_NAME, canvas)
        key = cv2.waitKey(30) & 0xFF
        if key in (ord('q'), ord('Q'), 27):
            return "quit"
        if key in (ord('n'), ord('N')):
            return "next"
    return "next"


# ─────────────────────────────────────────────
#  MAIN: Predict one video
# ─────────────────────────────────────────────
def predict_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[WARN] Cannot open: {video_path}")
        return "next"

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS) or 25
    delay        = max(1, int(1000 / fps))

    fight_count = 0
    pred_count  = 0
    frame_no    = 0
    label       = "NonFight"
    confidence  = 0.0
    paused      = False

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, DISPLAY_W, DISPLAY_H)

    print(f"\n[VIDEO] {os.path.basename(video_path)}  ({total_frames} frames @ {fps:.0f}fps)")

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break
            frame_no += 1

            # ── predict every SKIP_FRAMES frames ──
            if frame_no % SKIP_FRAMES == 0:
                img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
                img = img.astype("float32") / 255.0
                img = np.expand_dims(img, axis=0)

                pred       = model.predict(img, verbose=0)[0]
                label      = classes[np.argmax(pred)]
                confidence = float(np.max(pred))
                pred_count += 1

                if label == "Fight":
                    fight_count += 1

        fight_ratio = fight_count / max(pred_count, 1)

        # resize frame for display
        display = cv2.resize(frame, (DISPLAY_W, DISPLAY_H))
        display = draw_hud(display, label, confidence,
                           fight_ratio, frame_no, total_frames, paused)

        cv2.imshow(WINDOW_NAME, display)

        key = cv2.waitKey(1 if paused else delay) & 0xFF

        if key in (ord('q'), ord('Q'), 27):        # Q / ESC → quit
            cap.release()
            return "quit"
        elif key in (ord('n'), ord('N')):           # N → next video
            break
        elif key in (ord('p'), ord('P')):           # P → pause/resume
            paused = not paused

    cap.release()

    # ── final verdict ──
    fight_ratio   = fight_count / max(pred_count, 1)
    final_label   = "Fight Detected" if fight_ratio > FIGHT_THRESH else "No Fight"
    verdict_color = "🔴" if final_label == "Fight Detected" else "🟢"

    print(f"  Result  : {verdict_color} {final_label}")
    print(f"  Stats   : {fight_count}/{pred_count} frames classified as Fight ({fight_ratio*100:.1f}%)")

    action = show_result_screen(os.path.basename(video_path),
                                final_label, fight_count, pred_count)
    return action


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────
if not os.path.isdir(VIDEO_FOLDER):
    print(f"[ERROR] Folder not found: {VIDEO_FOLDER}")
    sys.exit(1)

video_files = sorted([
    os.path.join(VIDEO_FOLDER, f)
    for f in os.listdir(VIDEO_FOLDER)
    if f.lower().endswith((".avi", ".mp4", ".mkv", ".mov"))
])

if not video_files:
    print(f"[ERROR] No video files found in '{VIDEO_FOLDER}'")
    sys.exit(1)

print(f"\n[INFO] Found {len(video_files)} videos in '{VIDEO_FOLDER}'")
print("[INFO] Controls:  P = Pause/Resume   N = Next video   Q = Quit\n")

for vpath in video_files:
    action = predict_video(vpath)
    if action == "quit":
        print("\n[INFO] Exiting on user request.")
        break

cv2.destroyAllWindows()
print("\n[INFO] All done.")