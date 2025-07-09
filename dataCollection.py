import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
import os

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2)

offset = 20
imgSize = 300

folder = "Data/B"  # Default folder, will change dynamically
counter = 0
current_letter = 'B'
capturing = False

# Ensure all letter folders exist
for letter in [chr(i) for i in range(ord('A'), ord('Z')+1)]:
    os.makedirs(f"Data/{letter}", exist_ok=True)

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
    if hands:
        # If two hands, get a bounding box that covers both
        if len(hands) == 2:
            x1, y1, w1, h1 = hands[0]['bbox']
            x2, y2, w2, h2 = hands[1]['bbox']
            x_min = min(x1, x2)
            y_min = min(y1, y2)
            x_max = max(x1 + w1, x2 + w2)
            y_max = max(y1 + h1, y2 + h2)
            x, y, w, h = x_min, y_min, x_max - x_min, y_max - y_min
        else:
            x, y, w, h = hands[0]['bbox']

        # Ensure crop coordinates are within image bounds
        img_h, img_w = img.shape[:2]
        x1_crop = max(x - offset, 0)
        y1_crop = max(y - offset, 0)
        x2_crop = min(x + w + offset, img_w)
        y2_crop = min(y + h + offset, img_h)
        if x1_crop >= x2_crop or y1_crop >= y2_crop:
            continue  # Skip if crop is invalid

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y1_crop:y2_crop, x1_crop:x2_crop]

        imgCropShape = imgCrop.shape
        if imgCropShape[0] == 0 or imgCropShape[1] == 0:
            continue  # Skip if crop is empty
        aspectRation = h / w

        if aspectRation > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize

        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    # Start capturing when letter key is pressed
    if ord('a') <= key <= ord('z'):
        current_letter = chr(key).upper()
        folder = f"Data/{current_letter}"
        print(f"Started capturing for letter: {current_letter}")
        capturing = True
    elif ord('A') <= key <= ord('Z'):
        current_letter = chr(key)
        folder = f"Data/{current_letter}"
        print(f"Started capturing for letter: {current_letter}")
        capturing = True
    # Stop capturing when 'q' is pressed
    elif key == ord('q'):
        capturing = False
        print("Stopped capturing.")
    # Save images continuously if capturing
    if capturing and hands:
        save_path = f'Data/{current_letter}/Image_{time.time()}.jpg'
        cv2.imwrite(save_path, imgWhite)
        counter += 1
        print(f"Saved {save_path} ({counter})")

