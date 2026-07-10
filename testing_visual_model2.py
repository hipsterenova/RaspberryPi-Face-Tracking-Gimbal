import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import os
import time
from collections import deque 

# --- 1. Constants and Model Loading ---
PROJECT_ROOT = "d:/python/ml_project"
MODEL_PATH = os.path.join(PROJECT_ROOT, "visual_model.keras")
IMG_WIDTH, IMG_HEIGHT = 64, 64

# --- Smoothing/Debounce Logic ---
TALK_CONFIRM_SEC = 0.5  
SILENCE_CONFIRM_SEC = 0.7 
PREDICTION_THRESHOLD = 0.5 
current_status = "NOT TALKING" 
prediction_history = deque(maxlen=100) 

# --- FPS Calculation ---
prev_time = 0
fps = 30 

# Load the trained model
print(f"Loading model from: {MODEL_PATH}")
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# --- 2. MediaPipe & OpenCV Initialization ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)
print("\nStarting webcam feed (with smoothing)... Press 'q' to quit.")

# --- 3. Main Loop ---
while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue

    # --- FPS Calculation ---
    curr_time = time.time()
    if (curr_time - prev_time) > 0:
        fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    
    # --- Convert buffer time (sec) to frames (int) ---
    talk_frames_needed = int(TALK_CONFIRM_SEC * fps)
    silence_frames_needed = int(SILENCE_CONFIRM_SEC * fps)
    
    # --- Resize history buffer if needed ---
    max_frames_needed = max(talk_frames_needed, silence_frames_needed, 1) # Ensure maxlen is at least 1
    if prediction_history.maxlen != max_frames_needed:
         prediction_history = deque(maxlen=max_frames_needed)
    
    # --- Get Model Prediction ---
    image_height, image_width, _ = image.shape
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False
    results = face_mesh.process(image_rgb)
    image.flags.writeable = True

    raw_prediction = 0 

    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]
        landmarks = face_landmarks.landmark

        # Crop the Face
        all_x = [l.x * image_width for l in landmarks]
        all_y = [l.y * image_height for l in landmarks]
        x_min = max(0, int(min(all_x)) - 30)
        x_max = min(image_width, int(max(all_x)) + 30)
        y_min = max(0, int(min(all_y)) - 30)
        y_max = min(image_height, int(max(all_y)) + 30)
        
        face_crop = image[y_min:y_max, x_min:x_max]

        if face_crop.size > 0:
            face_resized = cv2.resize(face_crop, (IMG_WIDTH, IMG_HEIGHT))
            face_normalized = face_resized / 255.0
            face_batch = np.expand_dims(face_normalized, axis=0)
            
            confidence = model.predict(face_batch, verbose=0)[0][0]
            
            if confidence > PREDICTION_THRESHOLD:
                raw_prediction = 1 # Talking
            else:
                raw_prediction = 0 # Not Talking

            box_color = (0, 255, 0) if current_status == "NOT TALKING" else (0, 0, 255)
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), box_color, 2)
            
    prediction_history.append(raw_prediction)

    # --- Apply Smoothing Logic ---
    if len(prediction_history) == prediction_history.maxlen:
        
        if current_status == "NOT TALKING":
            talk_sum = sum(list(prediction_history)[-talk_frames_needed:])
            
            # --- THIS IS THE FIX ---
            if talk_frames_needed > 0: # Check before dividing
                talk_ratio = talk_sum / talk_frames_needed
                if talk_ratio > 0.90: 
                    current_status = "TALKING"
                
        elif current_status == "TALKING":
            silence_sum = sum(list(prediction_history)[-silence_frames_needed:])
            
            # --- THIS IS THE FIX ---
            if silence_frames_needed > 0: # Check before dividing
                silence_ratio = silence_sum / silence_frames_needed
                if silence_ratio < 0.30: 
                    current_status = "NOT TALKING"

    # --- 4. Display Status & Flip Image ---
    image = cv2.flip(image, 1) 
    
    color = (0, 0, 255) if current_status == "TALKING" else (0, 255, 0)
    
    cv2.putText(image, current_status, (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
    
    cv2.putText(image, f"FPS: {int(fps)}", (30, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow('Visual Model Test (Smoothed)', image)

    # --- 5. Key Press Logic ---
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- 6. Cleanup ---
print("Shutting down...")
cap.release()
cv2.destroyAllWindows()
face_mesh.close()