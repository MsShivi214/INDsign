import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import tensorflow as tf
import os

# --- CONFIGURATION ---
MODEL_PATH = "converted_keras_1/keras_model.h5"
LABELS_PATH = "converted_keras_1/labels.txt"
OFFSET = 20
STABLE_REQUIRED = 7  # Frames required for stable prediction

# --- MODEL LOADER ---
class CustomClassifier:
    def __init__(self, model_path, labels_path):
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.labels = []
        self.load_model(model_path)
        self.load_labels(labels_path)

    def load_model(self, model_path):
        if model_path.endswith('.h5'):
            tflite_path = model_path.replace('.h5', '.tflite')
        else:
            tflite_path = model_path
        if not os.path.exists(tflite_path):
            raise FileNotFoundError(f"TFLite model not found at {tflite_path}")
        self.interpreter = tf.lite.Interpreter(model_path=tflite_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        print(f"Model loaded from {tflite_path}")
        print(f"Input shape: {self.input_details[0]['shape']}")

    def load_labels(self, labels_path):
        with open(labels_path, 'r') as f:
            self.labels = []
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) >= 2:
                    self.labels.append(parts[1])
        print(f"Loaded {len(self.labels)} labels: {self.labels}")

    def getPrediction(self, img):
        input_shape = self.input_details[0]['shape']
        target_height, target_width = input_shape[1], input_shape[2]
        img = cv2.resize(img, (target_width, target_height))
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)
        self.interpreter.set_tensor(self.input_details[0]['index'], img)
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        prediction = output_data[0]
        index = np.argmax(prediction)
        confidence = prediction[index]
        return prediction, index, confidence

# --- MAIN SCRIPT ---
if not os.path.exists(MODEL_PATH) or not os.path.exists(LABELS_PATH):
    print(f"Error: Model files not found. Please ensure {MODEL_PATH} and {LABELS_PATH} exist.")
    exit()

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2, detectionCon=0.8)

try:
    classifier = CustomClassifier(MODEL_PATH, LABELS_PATH)
    input_shape = classifier.input_details[0]['shape']
    input_height, input_width = input_shape[1], input_shape[2]
except Exception as e:
    print(f"Error initializing classifier: {str(e)}")
    exit()

generated_text = ""
stable_letters = [None, None]
stable_counts = [0, 0]

print("Starting sign-to-text system...")
print("Show both hands in frame. Output will be generated automatically.")

while True:
    success, img = cap.read()
    if not success:
        print("Failed to grab frame")
        continue
    hands, img = detector.findHands(img)
    if hands:
        for i, hand in enumerate(hands[:2]):
            x, y, w, h = hand['bbox']
            img_h, img_w = img.shape[:2]
            x1_crop = max(x - OFFSET, 0)
            y1_crop = max(y - OFFSET, 0)
            x2_crop = min(x + w + OFFSET, img_w)
            y2_crop = min(y + h + OFFSET, img_h)
            if x1_crop >= x2_crop or y1_crop >= y2_crop:
                continue
            imgCrop = img[y1_crop:y2_crop, x1_crop:x2_crop]
            if imgCrop.shape[0] == 0 or imgCrop.shape[1] == 0:
                continue
            # Resize and center on white background
            imgWhite = np.ones((input_height, input_width, 3), np.uint8) * 255
            aspectRatio = h / w
            if aspectRatio > 1:
                k = input_height / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, input_height))
                wGap = math.ceil((input_width - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap] = imgResize
            else:
                k = input_width / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (input_width, hCal))
                hGap = math.ceil((input_height - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize
            # Convert BGR to RGB for model
            imgWhite_rgb = cv2.cvtColor(imgWhite, cv2.COLOR_BGR2RGB)
            prediction, index, confidence = classifier.getPrediction(imgWhite_rgb)
            predicted_letter = classifier.labels[index]
            # Draw bounding box and prediction
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)
            cv2.putText(img, f"{predicted_letter} {confidence:.2f}", (x, y-20), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 2)
            # Show cropped hand for debugging
            cv2.imshow(f"Hand {i+1} Crop", imgWhite)
            # Stability logic per hand
            if predicted_letter == stable_letters[i]:
                stable_counts[i] += 1
            else:
                stable_letters[i] = predicted_letter
                stable_counts[i] = 1
            # Only append if stable for required frames and not repeated
            if stable_counts[i] == STABLE_REQUIRED and (not generated_text or generated_text[-1] != stable_letters[i]):
                generated_text += stable_letters[i]
    # Display the generated text on the main image
    cv2.putText(img, f"Text: {generated_text}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(img, f"Hands: {len(hands) if hands else 0}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == 32:
        generated_text += ' '
    elif key == ord('c'):
        generated_text = ''
cap.release()
cv2.destroyAllWindows() 