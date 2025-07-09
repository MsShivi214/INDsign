import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
import tensorflow as tf
import os

# Custom TFLite model loader class
class CustomClassifier:
    def __init__(self, model_path, labels_path):
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.labels = []
        self.load_model(model_path)
        self.load_labels(labels_path)

    def load_model(self, model_path):
        try:
            # Use the existing TFLite model directly
            if model_path.endswith('.h5'):
                tflite_path = model_path.replace('.h5', '.tflite')
            else:
                tflite_path = model_path
                
            if not os.path.exists(tflite_path):
                print(f"TFLite model not found at {tflite_path}")
                raise FileNotFoundError(f"TFLite model not found")
            
            # Load the TFLite model
            self.interpreter = tf.lite.Interpreter(model_path=tflite_path)
            self.interpreter.allocate_tensors()
            
            # Get input and output details
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            
            print(f"Model loaded successfully from {tflite_path}")
            print(f"Input shape: {self.input_details[0]['shape']}")
            print(f"Output shape: {self.output_details[0]['shape']}")
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise

    def load_labels(self, labels_path):
        try:
            with open(labels_path, 'r') as f:
                # Parse labels in format "0 A", "1 B", etc.
                self.labels = []
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        self.labels.append(parts[1])  # Take the letter part
            print(f"Loaded {len(self.labels)} labels: {self.labels}")
        except Exception as e:
            print(f"Error loading labels: {str(e)}")
            raise

    def getPrediction(self, img):
        try:
            # Get the expected input shape from the model
            input_shape = self.input_details[0]['shape']
            target_height, target_width = input_shape[1], input_shape[2]
            
            # Preprocess the image to match training data format
            img = cv2.resize(img, (target_width, target_height))
            img = img.astype(np.float32) / 255.0
            
            # Add batch dimension
            img = np.expand_dims(img, axis=0)

            # Set the input tensor
            self.interpreter.set_tensor(self.input_details[0]['index'], img)
            
            # Run inference
            self.interpreter.invoke()
            
            # Get the output tensor
            output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
            
            # Process the results
            prediction = output_data[0]
            
            # Get top 3 predictions for debugging
            top_indices = np.argsort(prediction)[-3:][::-1]
            top_predictions = [(self.labels[i], prediction[i]) for i in top_indices]
            
            index = np.argmax(prediction)
            confidence = prediction[index]
            
            return prediction, index, confidence, top_predictions
            
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            return None, None, None, None

# Check if model files exist
model_path = "converted_keras_1/keras_model.h5"
labels_path = "converted_keras_1/labels.txt"

if not os.path.exists(model_path) or not os.path.exists(labels_path):
    print(f"Error: Model files not found. Please ensure {model_path} and {labels_path} exist.")
    exit()

# Initialize camera and detector
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2, detectionCon=0.8)

try:
    print("Loading model...")
    classifier = CustomClassifier(model_path, labels_path)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error initializing classifier: {str(e)}")
    exit()

offset = 20
imgSize = 300

print("Starting hand gesture recognition...")
print("Press 'q' to quit")
print("Detecting both hands for sign language recognition...")

# Variables for prediction smoothing
prediction_history = []
history_size = 5

# Letter-specific confidence thresholds based on model performance
confidence_thresholds = {
    'A': 0.3, 'B': 0.3, 'C': 0.3, 'D': 0.2, 'E': 0.3, 'F': 0.25, 'G': 0.3,
    'H': 0.3, 'I': 0.3, 'J': 0.25, 'K': 0.25, 'L': 0.3, 'M': 0.3, 'N': 0.2,
    'O': 0.3, 'P': 0.25, 'Q': 0.3, 'R': 0.3, 'S': 0.3, 'T': 0.3, 'U': 0.3,
    'V': 0.25, 'W': 0.3, 'X': 0.3, 'Y': 0.3, 'Z': 0.3
}

# Problematic letter pairs that are often confused
confusion_pairs = {
    'D': ['C', 'P', 'U'],  # D is often confused with C, P, U
    'N': ['M'],            # N is often confused with M
    'F': ['I', 'C', 'Z'],  # F is often confused with I, C, Z
    'J': ['X', 'A'],       # J is often confused with X, A
    'K': ['C', 'T'],       # K is often confused with C, T
    'V': ['I', 'M'],       # V is often confused with I, M
    'P': ['C']             # P is often confused with C
}

while True:
    success, img = cap.read()
    if not success:
        print("Failed to grab frame")
        continue
        
    hands, img = detector.findHands(img)
    
    if hands:
        # Handle both hands like in data collection
        if len(hands) == 2:
            # Get bounding box that covers both hands
            x1, y1, w1, h1 = hands[0]['bbox']
            x2, y2, w2, h2 = hands[1]['bbox']
            x_min = min(x1, x2)
            y_min = min(y1, y2)
            x_max = max(x1 + w1, x2 + w2)
            y_max = max(y1 + h1, y2 + h2)
            x, y, w, h = x_min, y_min, x_max - x_min, y_max - y_min
        else:
            # Single hand
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

        try:
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

            # Make prediction using the processed white image
            prediction, index, confidence, top_predictions = classifier.getPrediction(imgWhite)
            if prediction is not None and index is not None:
                predicted_letter = classifier.labels[index]
                
                # Add to prediction history for smoothing
                prediction_history.append((index, confidence))
                if len(prediction_history) > history_size:
                    prediction_history.pop(0)
                
                # Get most common prediction in recent history
                if len(prediction_history) >= 2:
                    recent_predictions = [p[0] for p in prediction_history[-3:]]
                    most_common = max(set(recent_predictions), key=recent_predictions.count)
                    avg_confidence = np.mean([p[1] for p in prediction_history if p[0] == most_common])
                    
                    # Get the threshold for this letter
                    threshold = confidence_thresholds.get(predicted_letter, 0.3)
                    
                    # Special handling for problematic letters
                    final_prediction = predicted_letter
                    final_confidence = avg_confidence
                    
                    # Check if this is a problematic letter and apply special logic
                    if predicted_letter in confusion_pairs:
                        # Look at top predictions to see if there's a better match
                        top_letter, top_conf = top_predictions[0]
                        second_letter, second_conf = top_predictions[1] if len(top_predictions) > 1 else (None, 0)
                        
                        # If the second prediction is in the confusion list and has similar confidence
                        if (second_letter in confusion_pairs[predicted_letter] and 
                            abs(top_conf - second_conf) < 0.1 and 
                            second_conf > threshold):
                            # Use the second prediction if it's more likely to be correct
                            final_prediction = second_letter
                            final_confidence = second_conf
                    
                    if final_confidence > threshold:
                        print(f"Predicted: {final_prediction} with confidence: {final_confidence:.2f}")
                        print(f"Top 3 predictions: {top_predictions}")
                        
                        # Display the prediction on the image
                        cv2.putText(img, f"{final_prediction} {final_confidence:.2f}", (x, y-20), 
                                  cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 2)
                        
                        # Draw bounding box
                        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)
                        
                        # Show number of hands detected
                        hand_count = len(hands)
                        cv2.putText(img, f"Hands: {hand_count}", (10, 30), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("ImageCrop", imgCrop)
            cv2.imshow("ImageWhite", imgWhite)

        except Exception as e:
            print(f"Error processing hand: {str(e)}")
            continue

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == ord('q'):  # Press 'q' to quit
        break

cap.release()
cv2.destroyAllWindows() 
