import cv2
import mediapipe as mp
import time
import math
import os 
import numpy as np 

# --- 1. Initialization ---

# --- Thresholds & Setup (USER UPDATES) ---
NOT_TALKING_THRESHOLD = 0.05  # Updated: More sensitive
TALKING_THRESHOLD = 0.2     # Updated: More sensitive
# Updated: Specific file path
DATA_PATH = "d:/python/ml_project/dataset" 
FOLDER_TALKING = os.path.join(DATA_PATH, "talking")
FOLDER_NOT_TALKING = os.path.join(DATA_PATH, "not_talking")
CROP_PADDING = 30 # Pixels to add around the face
is_recording = False # Toggle with 'r' key

# --- Create Folders ---
os.makedirs(FOLDER_TALKING, exist_ok=True)
os.makedirs(FOLDER_NOT_TALKING, exist_ok=True)
# This will now print the correct, full path
print(f"Dataset folders created at: {os.path.abspath(DATA_PATH)}")

# --- MediaPipe Setup ---
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1, # Only track one face for data collection
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# --- OpenCV Webcam Setup ---
cap = cv2.VideoCapture(0)
print("\nStarting webcam feed...")
print("Press 'r' to toggle RECORDING.")
print("Press 'q' to quit.")

# --- 2. Main Loop ---
while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue

    # Get image dimensions
    image_height, image_width, _ = image.shape

    # Process with MediaPipe
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False # Performance
    results = face_mesh.process(image_rgb)
    image_rgb.flags.writeable = True 

    lip_ratio = 0.0

    # --- 3. Per-Face Logic ---
    if results.multi_face_landmarks:
        # We set max_num_faces=1, so just use the first one
        face_landmarks = results.multi_face_landmarks[0]
        landmarks = face_landmarks.landmark
        
        # --- A. Calculate Lip Ratio ---
        lip_top_inner = landmarks[13]
        lip_bottom_inner = landmarks[14]
        mouth_left_corner = landmarks[78]
        mouth_right_corner = landmarks[308]

        lip_gap_y = abs(lip_top_inner.y - lip_bottom_inner.y)
        mouth_width_x = abs(mouth_right_corner.x - mouth_left_corner.x)

        if mouth_width_x > 0.0001: 
            lip_ratio = lip_gap_y / mouth_width_x
        
        # --- B. Save Frame if Recording ---
        if is_recording:
            # First, find the bounding box of the face
            all_x = [l.x * image_width for l in landmarks]
            all_y = [l.y * image_height for l in landmarks]
            
            x_min = int(min(all_x)) - CROP_PADDING
            x_max = int(max(all_x)) + CROP_PADDING
            y_min = int(min(all_y)) - CROP_PADDING
            y_max = int(max(all_y)) + CROP_PADDING

            # Clamp values to be inside the image
            x_min = max(0, x_min)
            x_max = min(image_width, x_max)
            y_min = max(0, y_min)
            y_max = min(image_height, y_max)
            
            # Crop the *original BGR* face image
            face_crop = image[y_min:y_max, x_min:x_max]

            # Generate a unique filename
            filename = f"{int(time.time() * 1000)}.png"

            # Save to the correct folder (using the new thresholds)
            if lip_ratio > TALKING_THRESHOLD:
                save_path = os.path.join(FOLDER_TALKING, filename)
                cv2.imwrite(save_path, face_crop)
            
            elif lip_ratio < NOT_TALKING_THRESHOLD:
                save_path = os.path.join(FOLDER_NOT_TALKING, filename)
                cv2.imwrite(save_path, face_crop)
                
        # --- C. Draw Contours for Visualization ---
        mp_drawing.draw_landmarks(
            image=image, # Draw on the BGR image
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles
            .get_default_face_mesh_contours_style())

    # --- 4. Display Status & Flip Image ---
    
    # Flip the image *first* for a "selfie" view
    image = cv2.flip(image, 1)
    
    # Display Lip Ratio
    cv2.putText(image, f"Lip Ratio: {lip_ratio:.4f}", (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Display Recording Status
    if is_recording:
        cv2.putText(image, "[RECORDING]", (30, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        cv2.putText(image, "[PAUSED]", (30, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Display the final image
    cv2.imshow('V1 Data Collection', image)

    # --- 5. Key Press Logic ---
    key = cv2.waitKey(5) & 0xFF
    
    if key == ord('q'):
        break
    
    if key == ord('r'):
        is_recording = not is_recording # Toggle recording
        print(f"Recording status: {is_recording}")

# --- 6. Cleanup ---
print("Shutting down...")
cap.release()
cv2.destroyAllWindows()
face_mesh.close()