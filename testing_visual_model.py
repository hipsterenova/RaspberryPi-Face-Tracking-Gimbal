import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import os

# --- 1. Constants and Model Loading ---
PROJECT_ROOT = "d:/python/ml_project"
MODEL_PATH = os.path.join(PROJECT_ROOT, "visual_model.keras")

# These MUST match the values from your training script
IMG_WIDTH = 64
IMG_HEIGHT = 64

# Load the trained model
print(f"Loading model from: {MODEL_PATH}")
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please ensure 'visual_model.keras' is in the correct directory.")
    exit()

# Threshold for prediction (0.5 is the default for sigmoid)
PREDICTION_THRESHOLD = 0.5

# --- 2. MediaPipe & OpenCV Initialization ---
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)
print("\nStarting webcam feed... Press 'q' to quit.")

# --- 3. Main Loop ---
while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue

    image_height, image_width, _ = image.shape
    
    # Process with MediaPipe
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False
    results = face_mesh.process(image_rgb)
    
    # We'll draw on the original BGR image
    image.flags.writeable = True

    # Default label
    label = "NOT TALKING"
    color = (0, 255, 0) # Green

    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]
        landmarks = face_landmarks.landmark

        # --- A. Crop the Face (same as data collector) ---
        all_x = [l.x * image_width for l in landmarks]
        all_y = [l.y * image_height for l in landmarks]
        
        x_min = int(min(all_x)) - 30
        x_max = int(max(all_x)) + 30
        y_min = int(min(all_y)) - 30
        y_max = int(max(all_y)) + 30

        x_min = max(0, x_min)
        x_max = min(image_width, x_max)
        y_min = max(0, y_min)
        y_max = min(image_height, y_max)
        
        face_crop = image[y_min:y_max, x_min:x_max]

        if face_crop.size > 0: # Check if crop is valid
            
            # --- B. Pre-process the Crop for the Model ---
            # 1. Resize to 64x64
            face_resized = cv2.resize(face_crop, (IMG_WIDTH, IMG_HEIGHT))
            # 2. Normalize
            face_normalized = face_resized / 255.0
            # 3. Expand dimensions to (1, 64, 64, 3)
            face_batch = np.expand_dims(face_normalized, axis=0)
            
            # --- C. Make Prediction ---
            prediction = model.predict(face_batch, verbose=0)
            confidence = prediction[0][0]

            # --- D. Set Label ---
            if confidence > PREDICTION_THRESHOLD:
                label = "TALKING"
                color = (0, 0, 255) # Red
            else:
                label = "NOT TALKING"
                color = (0, 255, 0) # Green

        # --- E. Draw Bounding Box (use the crop coordinates) ---
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)

    # --- 4. Display Status & Flip Image ---
    
    # Flip the image *first* for a "selfie" view
    image = cv2.flip(image, 1)
    
    # Put the final label on the flipped image
    cv2.putText(image, label, (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 2)

    # Display the final image
    cv2.imshow('Visual Model Test', image)

    # --- 5. Key Press Logic ---
    key = cv2.waitKey(5) & 0xFF
    if key == ord('q'):
        break

# --- 6. Cleanup ---
print("Shutting down...")
cap.release()
cv2.destroyAllWindows()
face_mesh.close()