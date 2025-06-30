import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
import math
from datetime import datetime, date
import os

cap = cv2.VideoCapture(0)
detector = HandDetector()

offset = 20
img_size = 300

folder = "data/Dekemvri"
os.makedirs(folder, exist_ok=True)
counter = 0

while True:
    success, img = cap.read()
    if not success:
        continue

    white_img = None  # INIT: empty for each frame

    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        white_img = np.ones((img_size, img_size, 3), np.uint8) * 255

        y1, y2 = max(0, y - offset), min(img.shape[0], y + h + offset)
        x1, x2 = max(0, x - offset), min(img.shape[1], x + w + offset)

        img_crop = img[y1:y2, x1:x2]

        if img_crop.size > 0:
            aspect_ratio = h / w

            try:
                if aspect_ratio > 1:
                    k = img_size / h
                    width_calculated = math.ceil(k * w)
                    if width_calculated > 0:
                        img_resize = cv2.resize(img_crop, (width_calculated, img_size))
                        width_gap = math.ceil((img_size - width_calculated) / 2)
                        white_img[:, width_gap:width_gap + width_calculated] = img_resize
                else:
                    k = img_size / w
                    height_calculated = math.ceil(k * h)
                    if height_calculated > 0:
                        img_resize = cv2.resize(img_crop, (img_size, height_calculated))
                        height_gap = math.ceil((img_size - height_calculated) / 2)
                        white_img[height_gap:height_gap + height_calculated, :] = img_resize
            except Exception as e:
                print(f"Resize error: {e}")

            cv2.imshow("Cropped Image", img_crop)
            cv2.imshow("White image", white_img)

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == ord("s") and white_img is not None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        file_path = f"{folder}/handgesture_{timestamp}.jpg"
        cv2.imwrite(file_path, white_img)
        counter += 1
        print(f"[{counter}] Saved to {file_path}")

