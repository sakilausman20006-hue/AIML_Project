import os
import cv2

INPUT_DIR = "train"
OUTPUT_DIR = "frames_train"

classes = ["fight","nonfight"]

for cls in classes:
    os.makedirs(os.path.join(OUTPUT_DIR, cls), exist_ok=True)

    class_path = os.path.join(INPUT_DIR, cls)

    for video in os.listdir(class_path):
        video_path = os.path.join(class_path, video)

        cap = cv2.VideoCapture(video_path)

        count = 0
        success, frame = cap.read()

        while success:
            frame = cv2.resize(frame,(64,64))

            save_path = os.path.join(
                OUTPUT_DIR,
                cls,
                f"{video}_{count}.jpg"
            )

            cv2.imwrite(save_path, frame)

            success, frame = cap.read()
            count += 10   # skip frames

        cap.release()

print("Frames extracted")