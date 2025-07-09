import cv2
import numpy as np
import tensorflow as tf
import os
import random
from collections import defaultdict

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

def test_model_on_training_data():
    # Initialize the classifier
    model_path = "converted_keras_1/keras_model.h5"
    labels_path = "converted_keras_1/labels.txt"
    
    if not os.path.exists(model_path) or not os.path.exists(labels_path):
        print(f"Error: Model files not found.")
        return
    
    try:
        print("Loading model...")
        classifier = CustomClassifier(model_path, labels_path)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error initializing classifier: {str(e)}")
        return
    
    # Test results storage
    results = defaultdict(list)
    letter_accuracy = {}
    
    # Test each letter
    for letter in classifier.labels:
        letter_folder = f"Data/{letter}"
        if not os.path.exists(letter_folder):
            print(f"Warning: No data folder found for letter {letter}")
            continue
        
        # Get list of image files
        image_files = [f for f in os.listdir(letter_folder) if f.endswith('.jpg')]
        if not image_files:
            print(f"Warning: No images found for letter {letter}")
            continue
        
        print(f"\nTesting letter {letter} with {len(image_files)} images...")
        
        # Test with a sample of images (max 10 per letter to avoid too much output)
        test_images = random.sample(image_files, min(10, len(image_files)))
        
        correct_predictions = 0
        total_predictions = 0
        
        for img_file in test_images:
            img_path = os.path.join(letter_folder, img_file)
            
            try:
                # Load the image
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Could not load image: {img_path}")
                    continue
                
                # Make prediction
                prediction, index, confidence, top_predictions = classifier.getPrediction(img)
                
                if prediction is not None and index is not None:
                    predicted_letter = classifier.labels[index]
                    is_correct = predicted_letter == letter
                    
                    if is_correct:
                        correct_predictions += 1
                    total_predictions += 1
                    
                    results[letter].append({
                        'file': img_file,
                        'predicted': predicted_letter,
                        'actual': letter,
                        'confidence': confidence,
                        'top_predictions': top_predictions,
                        'correct': is_correct
                    })
                    
                    # Print detailed results for incorrect predictions
                    if not is_correct:
                        print(f"  ❌ {img_file}: Predicted {predicted_letter} (conf: {confidence:.3f}), Actual: {letter}")
                        print(f"     Top 3: {top_predictions}")
                    else:
                        print(f"  ✅ {img_file}: Correctly predicted {predicted_letter} (conf: {confidence:.3f})")
                
            except Exception as e:
                print(f"Error processing {img_file}: {str(e)}")
        
        # Calculate accuracy for this letter
        if total_predictions > 0:
            accuracy = correct_predictions / total_predictions
            letter_accuracy[letter] = accuracy
            print(f"Letter {letter} accuracy: {accuracy:.2%} ({correct_predictions}/{total_predictions})")
        else:
            letter_accuracy[letter] = 0.0
            print(f"Letter {letter} accuracy: No valid predictions")
    
    # Print summary
    print("\n" + "="*50)
    print("MODEL ACCURACY SUMMARY")
    print("="*50)
    
    # Sort letters by accuracy
    sorted_letters = sorted(letter_accuracy.items(), key=lambda x: x[1], reverse=True)
    
    print("\nLetters with good recognition (>80% accuracy):")
    for letter, accuracy in sorted_letters:
        if accuracy > 0.8:
            print(f"  {letter}: {accuracy:.2%}")
    
    print("\nLetters with poor recognition (<50% accuracy):")
    for letter, accuracy in sorted_letters:
        if accuracy < 0.5:
            print(f"  {letter}: {accuracy:.2%}")
    
    print("\nLetters with moderate recognition (50-80% accuracy):")
    for letter, accuracy in sorted_letters:
        if 0.5 <= accuracy <= 0.8:
            print(f"  {letter}: {accuracy:.2%}")
    
    # Overall accuracy
    overall_accuracy = sum(letter_accuracy.values()) / len(letter_accuracy) if letter_accuracy else 0
    print(f"\nOverall model accuracy: {overall_accuracy:.2%}")
    
    # Detailed analysis for problematic letters
    print("\n" + "="*50)
    print("DETAILED ANALYSIS FOR PROBLEMATIC LETTERS")
    print("="*50)
    
    for letter, accuracy in sorted_letters:
        if accuracy < 0.5:
            print(f"\nLetter {letter} (accuracy: {accuracy:.2%}):")
            letter_results = results[letter]
            
            # Show what the model is predicting instead
            prediction_counts = defaultdict(int)
            for result in letter_results:
                prediction_counts[result['predicted']] += 1
            
            print("  Model predictions:")
            for pred, count in sorted(prediction_counts.items(), key=lambda x: x[1], reverse=True):
                print(f"    {pred}: {count} times")
            
            # Show some example top predictions
            print("  Sample top predictions:")
            for result in letter_results[:3]:
                print(f"    {result['file']}: {result['top_predictions']}")

if __name__ == "__main__":
    test_model_on_training_data() 