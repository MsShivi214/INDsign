# INDsign 

# INDsign: Real-Time Sign Language Recognition

This project uses a webcam and a trained TensorFlow Lite model to recognize sign language letters from one or both hands in real time, and generates text from the recognized signs.

## Features

- **Real-time hand detection** using OpenCV and cvzone.
- **Simultaneous recognition of both hands** (each hand is recognized independently).
- **Automatic text generation** from stable predictions.
- **Debug windows** show the cropped hand images being sent to the model.
- **Easy-to-use interface**: just run the script and start signing!

---

## Directory Structure

```
INDsign/
├── converted_keras_1/
│   ├── keras_model.h5
│   ├── keras_model.tflite
│   └── labels.txt
├── Data/                # Raw data (images) for each sign
├── Improved_Data/       # Improved/augmented data (if any)
├── sample_analysis/     # Sample images for analysis
├── analyze_training_data.py
├── dataCollection.py
├── retrain_model.py
├── test_model_accuracy.py
├── test.py
├── sign_to_text.py      # Main script for real-time sign-to-text
```

---

## Requirements

Install the following Python packages (preferably in a virtual environment):

```sh
pip install opencv-python cvzone numpy tensorflow
```

- **cvzone**: For hand detection (uses mediapipe internally)
- **opencv-python**: For image processing and webcam access
- **numpy**: For array operations
- **tensorflow**: For running the TFLite model

> **Note:**  
> If you have a GPU and want to use it for training, install `tensorflow-gpu`.  
> For inference, regular `tensorflow` is sufficient.

---

## Model Files

- Place your trained model files in `converted_keras_1/`:
  - `keras_model.tflite` (TensorFlow Lite model)
  - `labels.txt` (labels file, one line per class, e.g., `0 A`, `1 B`, ...)

---

## How to Run

1. **Connect your webcam.**
2. **Open a terminal in the project root.**
3. **Run the main script:**

   ```sh
   python sign_to_text.py
   ```

4. **Instructions:**
   - Show one or both hands in the frame.
   - The script will display:
     - The main video window with bounding boxes and predictions for each hand.
     - Cropped hand images for debugging.
     - The generated text at the top of the main window.
   - **Controls:**
     - Press `q` to quit.
     - Press `space` to add a space to the text.
     - Press `c` to clear the generated text.


---

## Training and Improving the Model

- Use `dataCollection.py` to collect more data.
- Use `retrain_model.py` to retrain your model with new data.
- Use `analyze_training_data.py` and `test_model_accuracy.py` to analyze and test your model.

---

## File Descriptions

- **sign_to_text.py**: Main script for real-time sign language recognition and text generation.
- **test.py**: (Legacy) Original test script for single-hand sign recognition.
- **dataCollection.py**: Script to collect training data using your webcam.
- **retrain_model.py**: Script to retrain the model with new or improved data.
- **analyze_training_data.py**: Analyze the distribution and quality of your training data.
- **test_model_accuracy.py**: Test the accuracy of your trained model on a test set.

---

