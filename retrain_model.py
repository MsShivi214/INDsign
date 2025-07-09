import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import os
import matplotlib.pyplot as plt
from collections import Counter

# Parameters
DATA_DIR = "Improved_Data"
BATCH_SIZE = 16
IMG_SIZE = (128, 128)
EPOCHS = 30
NUM_CLASSES = 26

# Data generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_generator = val_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# Model definition
model = models.Sequential([
    layers.Input(shape=(*IMG_SIZE, 3)),
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(NUM_CLASSES, activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# Callbacks
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=10,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7
    ),
    tf.keras.callbacks.ModelCheckpoint(
        'best_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

# Training
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    callbacks=callbacks,
    verbose=1
)

# Evaluation
val_generator.reset()
val_steps = val_generator.samples // val_generator.batch_size + 1
preds = model.predict(val_generator, steps=val_steps, verbose=1)
y_true = val_generator.classes
y_pred = preds.argmax(axis=1)

from sklearn.metrics import classification_report
labels = list(val_generator.class_indices.keys())
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=labels))

# Save model and labels
os.makedirs('converted_keras_2', exist_ok=True)
model.save('converted_keras_2/keras_model.h5')
with open('converted_keras_2/labels.txt', 'w') as f:
    for i, label in enumerate(labels):
        f.write(f"{i} {label}\n")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open('converted_keras_2/keras_model.tflite', 'wb') as f:
    f.write(tflite_model)
print("Model and labels saved in 'converted_keras_2'.")

# Plot training history
try:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    plt.tight_layout()
    plt.savefig('training_history.png')
    print("Training history plot saved as 'training_history.png'")
except Exception as e:
    print(f"Could not create training plot: {e}")

def load_and_preprocess_data(data_dir="Improved_Data", target_size=(128, 128)):
    """Load and preprocess all training data with reduced memory usage."""
    
    print("Loading and preprocessing training data...")
    
    X = []  # Images
    y = []  # Labels
    label_to_index = {}
    index_to_label = {}
    
    letters = [chr(i) for i in range(ord('A'), ord('Z')+1)]
    
    for i, letter in enumerate(letters):
        label_to_index[letter] = i
        index_to_label[i] = letter
        
        letter_dir = os.path.join(data_dir, letter)
        if not os.path.exists(letter_dir):
            print(f"Warning: No data folder for letter {letter}")
            continue
            
        image_files = [f for f in os.listdir(letter_dir) if f.endswith('.jpg')]
        
        print(f"Loading {len(image_files)} images for letter {letter}")
        
        # Process images in smaller batches to reduce memory usage
        batch_size = 50
        for j in range(0, len(image_files), batch_size):
            batch_files = image_files[j:j+batch_size]
            
            for img_file in batch_files:
                img_path = os.path.join(letter_dir, img_file)
                img = cv2.imread(img_path)
                
                if img is not None:
                    # Resize to target size (smaller to save memory)
                    img = cv2.resize(img, target_size)
                    
                    # Normalize pixel values
                    img = img.astype(np.float32) / 255.0
                    
                    X.append(img)
                    y.append(i)
            
            # Clear some memory
            if j % 100 == 0:
                print(f"  Processed {j+len(batch_files)} images for {letter}")
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"Loaded {len(X)} images with {len(set(y))} classes")
    print(f"Image shape: {X.shape}")
    
    return X, y, label_to_index, index_to_label

def create_improved_model(input_shape=(128, 128, 3), num_classes=26):
    """Create an improved CNN model for sign language recognition with smaller size."""
    
    model = models.Sequential([
        # Input layer
        layers.Input(shape=input_shape),
        
        # First convolutional block
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Second convolutional block
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Third convolutional block
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Fourth convolutional block
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Flatten and dense layers
        layers.Flatten(),
        layers.Dense(256, activation='relu'),  # Reduced from 512
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),  # Reduced from 256
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def train_model(X, y, label_to_index, index_to_label):
    """Train the improved model."""
    
    print("Preparing training data...")
    
    # Split data into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {len(X_train)} images")
    print(f"Validation set: {len(X_val)} images")
    
    # Convert labels to categorical
    y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes=26)
    y_val_cat = tf.keras.utils.to_categorical(y_val, num_classes=26)
    
    # Create data augmentation for training
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=False,  # Don't flip sign language images
        fill_mode='nearest'
    )
    
    # Create the model
    print("Creating improved model...")
    model = create_improved_model()
    
    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Print model summary
    model.summary()
    
    # Create callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7
        ),
        tf.keras.callbacks.ModelCheckpoint(
            'best_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Train the model with smaller batch size
    print("Training model...")
    history = model.fit(
        datagen.flow(X_train, y_train_cat, batch_size=16),  # Reduced batch size
        epochs=30,  # Reduced epochs
        validation_data=(X_val, y_val_cat),
        callbacks=callbacks,
        verbose=1
    )
    
    return model, history, X_val, y_val

def evaluate_model(model, X_val, y_val, index_to_label):
    """Evaluate the trained model."""
    
    print("\nEvaluating model...")
    
    # Make predictions in batches to save memory
    batch_size = 32
    y_pred_classes = []
    
    for i in range(0, len(X_val), batch_size):
        batch_X = X_val[i:i+batch_size]
        y_pred_batch = model.predict(batch_X, verbose=0)
        y_pred_classes.extend(np.argmax(y_pred_batch, axis=1))
    
    y_pred_classes = np.array(y_pred_classes)
    
    # Calculate accuracy
    accuracy = np.mean(y_pred_classes == y_val)
    print(f"Overall accuracy: {accuracy:.4f}")
    
    # Per-class accuracy
    print("\nPer-class accuracy:")
    class_accuracy = {}
    for i in range(26):
        mask = y_val == i
        if np.sum(mask) > 0:
            class_acc = np.mean(y_pred_classes[mask] == y_val[mask])
            class_accuracy[index_to_label[i]] = class_acc
            print(f"{index_to_label[i]}: {class_acc:.4f}")
    
    # Identify problematic classes
    problematic_classes = [letter for letter, acc in class_accuracy.items() if acc < 0.8]
    if problematic_classes:
        print(f"\nProblematic classes (<80% accuracy): {problematic_classes}")
    
    return class_accuracy

def save_model_and_labels(model, label_to_index, output_dir="converted_keras_2"):
    """Save the trained model and labels."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the model
    model_path = os.path.join(output_dir, "keras_model.h5")
    model.save(model_path)
    print(f"Model saved to {model_path}")
    
    # Save labels
    labels_path = os.path.join(output_dir, "labels.txt")
    with open(labels_path, 'w') as f:
        for i in range(26):
            letter = chr(ord('A') + i)
            f.write(f"{i} {letter}\n")
    print(f"Labels saved to {labels_path}")
    
    # Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    
    tflite_path = os.path.join(output_dir, "keras_model.tflite")
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    print(f"TFLite model saved to {tflite_path}")

def plot_training_history(history):
    """Plot training history."""
    
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot accuracy
        ax1.plot(history.history['accuracy'], label='Training Accuracy')
        ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        
        # Plot loss
        ax2.plot(history.history['loss'], label='Training Loss')
        ax2.plot(history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig('training_history.png')
        print("Training history plot saved as 'training_history.png'")
    except Exception as e:
        print(f"Could not create training plot: {e}")

if __name__ == "__main__":
    print("="*60)
    print("RETRAINING SIGN LANGUAGE RECOGNITION MODEL")
    print("="*60)
    
    # Load and preprocess data
    X, y, label_to_index, index_to_label = load_and_preprocess_data()
    
    # Train the model
    model, history, X_val, y_val = train_model(X, y, label_to_index, index_to_label)
    
    # Evaluate the model
    class_accuracy = evaluate_model(model, X_val, y_val, index_to_label)
    
    # Save the model
    save_model_and_labels(model, label_to_index)
    
    # Plot training history
    plot_training_history(history)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print("Model saved in 'converted_keras_2' folder")
    print("Use the new model for better recognition accuracy") 