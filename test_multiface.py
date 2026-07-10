import cv2
import keras
import mediapipe as mp
import numpy as np
import tensorflow as tf
import os
from collections import OrderedDict  # <-- Still need this for the tracker
from scipy.spatial import distance as dist 

# --- CentroidTracker Class (Unchanged) ---
class CentroidTracker:
    def __init__(self, maxDisappeared=30): 
        self.nextFaceID = 0
        self.faces = OrderedDict()
        self.disappeared = OrderedDict()
        self.maxDisappeared = maxDisappeared

    def register(self, centroid):
        self.faces[self.nextFaceID] = centroid
        self.disappeared[self.nextFaceID] = 0
        self.nextFaceID += 1

    def deregister(self, faceID):
        del self.faces[faceID]
        del self.disappeared[faceID]

    def update(self, rects):
        if len(rects) == 0:
            for faceID in list(self.disappeared.keys()):
                self.disappeared[faceID] += 1
                if self.disappeared[faceID] > self.maxDisappeared:
                    self.deregister(faceID)
            return self.faces

        inputCentroids = np.zeros((len(rects), 2), dtype="int")
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)

        if len(self.faces) == 0:
            for i in range(len(inputCentroids)):
                self.register(inputCentroids[i])
            return self.faces

        faceIDs = list(self.faces.keys())
        faceCentroids = list(self.faces.values())
        D = dist.cdist(np.array(faceCentroids), inputCentroids)
        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        usedRows = set()
        usedCols = set()
        for (row, col) in zip(rows, cols):
            if row in usedRows or col in usedCols:
                continue
            faceID = faceIDs[row]
            self.faces[faceID] = inputCentroids[col]
            self.disappeared[faceID] = 0
            usedRows.add(row)
            usedCols.add(col)

        unusedRows = set(range(D.shape[0])).difference(usedRows)
        unusedCols = set(range(D.shape[1])).difference(usedCols)

        if D.shape[0] >= D.shape[1]:
            for row in unusedRows:
                faceID = faceIDs[row]
                self.disappeared[faceID] += 1
                if self.disappeared[faceID] > self.maxDisappeared:
                    self.deregister(faceID)
        else:
            for col in unusedCols:
                self.register(inputCentroids[col])
        return self.faces

# --- 1. Constants and Model Loading ---
PROJECT_ROOT = "d:/python/ml_project"
MODEL_PATH = os.path.join(PROJECT_ROOT, "visual_model.keras")
IMG_WIDTH, IMG_HEIGHT = 64, 64
PREDICTION_THRESHOLD = 0.5 # Confidence threshold

# --- REMOVED: All smoothing/FPS/state variables ---

ct = CentroidTracker() # Initialize the tracker

# Load the trained model
print(f"Loading model from: {MODEL_PATH}")
model = keras.models.load_model(MODEL_PATH)
print("Model loaded successfully.")

# --- 2. MediaPipe & OpenCV Initialization ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=5, # Detect up to 5 faces
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)
print("\nStarting webcam feed (Multi-Face, Instant)... Press 'q' to quit.")

# --- 3. Main Loop ---
while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue
        
    # --- Text-Flip Fix: Flip the image *first* ---
    image = cv2.flip(image, 1)

    # --- REMOVED: FPS Calculation ---

    # --- Get Face Detections ---
    image_height, image_width, _ = image.shape
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
    image_rgb.flags.writeable = False
    results = face_mesh.process(image_rgb)
    image.flags.writeable = True

    # --- Prepare data for tracker ---
    current_frame_rects = []
    centroid_to_landmarks_map = {} 
    
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Get Bounding Box
            landmarks = face_landmarks.landmark
            all_x = [l.x * image_width for l in landmarks]
            all_y = [l.y * image_height for l in landmarks]
            x_min = max(0, int(min(all_x)) - 30)
            x_max = min(image_width, int(max(all_x)) + 30)
            y_min = max(0, int(min(all_y)) - 30)
            y_max = min(image_height, int(max(all_y)) + 30)
            
            rect = (x_min, y_min, x_max, y_max)
            current_frame_rects.append(rect)
            
            # Get Centroid
            cX = int((x_min + x_max) / 2.0)
            cY = int((y_min + y_max) / 2.0)
            centroid_to_landmarks_map[(cX, cY)] = (face_landmarks, rect)

    # --- Update Tracker ---
    tracked_faces = ct.update(current_frame_rects)

    # --- Main Logic Loop (Simplified: No State/Buffer) ---
    for (face_ID, centroid) in tracked_faces.items():
        
        # Look up the face data using the centroid
        if tuple(centroid) in centroid_to_landmarks_map:
            face_landmarks, (x_min, y_min, x_max, y_max) = centroid_to_landmarks_map[tuple(centroid)]
            
            # --- Run Prediction ---
            face_crop = image[y_min:y_max, x_min:x_max]
            
            # Set defaults
            label = "NOT TALKING"
            color = (0, 255, 0) # Green

            if face_crop.size > 0:
                face_resized = cv2.resize(face_crop, (IMG_WIDTH, IMG_HEIGHT))
                face_normalized = face_resized / 255.0
                face_batch = np.expand_dims(face_normalized, axis=0)
                
                # Make instant prediction
                confidence = model.predict(face_batch, verbose=0)[0][0]
                
                # Update label based on instant prediction
                if confidence > PREDICTION_THRESHOLD:
                    label = "TALKING"
                    color = (0, 0, 255) # Red
            
            # --- REMOVED: Smoothing Logic ---

            # --- Draw on Frame ---
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)
            cv2.putText(image, f"Person {face_ID}: {label}", 
                        (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.7, color, 2)

    # --- REMOVED: Stale Face ID Cleanup ---

    # --- 4. Display Image ---
    cv2.imshow('Visual Model Test (Multi-Face, Instant)', image)

    # --- 5. Key Press Logic ---
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- 6. Cleanup ---
print("Shutting down...")
cap.release()
cv2.destroyAllWindows()
face_mesh.close()