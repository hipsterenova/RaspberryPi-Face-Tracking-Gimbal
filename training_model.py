import cv2
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# --- 1. Constants (PATH FIX) ---
PROJECT_ROOT = "d:/python/ml_project" # <--- YOUR PATH
DATA_PATH = os.path.join(PROJECT_ROOT, "dataset")
IMG_WIDTH = 64  
IMG_HEIGHT = 64
EPOCHS = 15      
BATCH_SIZE = 32

# --- 2. Load and Preprocess Data ---
def load_data(data_dir):
    images = []
    labels = []
    
    # Load 'not_talking' (Label 0)
    not_talking_dir = os.path.join(data_dir, "not_talking")
    if not os.path.exists(not_talking_dir):
        print(f"Warning: Directory not found {not_talking_dir}")
        return None, None
        
    for filename in os.listdir(not_talking_dir):
        img_path = os.path.join(not_talking_dir, filename)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
            img = img / 255.0  # Normalize to [0, 1]
            images.append(img)
            labels.append(0)

    # Load 'talking' (Label 1)
    talking_dir = os.path.join(data_dir, "talking")
    if not os.path.exists(talking_dir):
        print(f"Warning: Directory not found {talking_dir}")
        return None, None

    for filename in os.listdir(talking_dir):
        img_path = os.path.join(talking_dir, filename)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
            img = img / 255.0  # Normalize to [0, 1]
            images.append(img)
            labels.append(1)
            
    if not images:
        return None, None

    # Convert to numpy arrays
    X = np.array(images)
    y = np.array(labels)
    
    return X, y

print("Loading dataset...")
X, y = load_data(DATA_PATH)

if X is None or len(X) == 0:
    print(f"Error: No images found in {DATA_PATH}. Please run data collection first.")
else:
    print(f"Loaded {len(X)} images.")

    # --- 3. Split Data ---
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")

    # --- 4. Define the CNN Model (Practical V1) ---
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
        MaxPooling2D((2, 2)),
        
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.5), 
        Dense(1, activation='sigmoid') 
    ])

    model.summary()

    # --- 5. Compile the Model ---
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    # --- 6. Train the Model ---
    print("\nStarting model training...")
    history = model.fit(
        X_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val)
    )
    print("Training complete.")

    # --- 7. Save the Model (PATH FIX) ---
    model_save_path = os.path.join(PROJECT_ROOT, 'visual_model.keras')
    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")

    # --- 8. Plot and Save Results (PATH FIX) ---
    plt.figure(figsize=(12, 4))
    
    # Plot Training & Validation Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot Training & Validation Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plot_save_path = os.path.join(PROJECT_ROOT, 'training_results.png')
    plt.savefig(plot_save_path)
    print(f"Training results plot saved to {plot_save_path}")