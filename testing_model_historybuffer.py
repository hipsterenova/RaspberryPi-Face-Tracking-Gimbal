import cv2
import keras
import mediapipe as mp
import numpy as np
import tensorflow as tf
import os
import time
from collections import OrderedDict
from scipy.spatial import distance as dist 

# --- TemporalSpeechDetector Class: MOTION-BASED FIX ---
class TemporalSpeechDetector:
    """
    Implements Motion-Based Temporal Smoothing Logic.
    
    Prediction is based on the VARIABILITY (switching between 0 and 1) 
    in the raw model output over a short window (0.5 seconds).
    
    Logic:
    - If frames are all '0' (closed) or all '1' (static open), the state is NOT TALKING (0).
    - If frames are mixed (e.g., 0, 1, 0, 1, 0), the state is TALKING (1).
    """
    def __init__(self, fps=30):
        self.FPS = fps
        # The window over which we check for continuous motion/switching
        self.MOTION_WINDOW_SECONDS = 0.5 
        self.MOTION_WINDOW_FRAMES = self.FPS * self.MOTION_WINDOW_SECONDS
        
        # We need enough history to check the full window
        self.MAX_HISTORY_FRAMES = self.MOTION_WINDOW_FRAMES 
        
        # State storage: { personId: { buffer: [0, 1, ...], is_talking: 0 or 1 } }
        self.person_states = {}

    def process_prediction(self, person_id, raw_prediction):
        if person_id not in self.person_states:
            self.person_states[person_id] = { 
                'buffer': [], 
                'is_talking': 0 # 0: Silent, 1: Talking (Stabilized State)
            }

        state = self.person_states[person_id]
        
        # 1. Update Buffer (Add new prediction)
        state['buffer'].append(raw_prediction)
        
        # Maintain buffer size exactly equal to the motion window
        if len(state['buffer']) > self.MOTION_WINDOW_FRAMES:
            state['buffer'].pop(0)

        # Ensure we have a full window before checking for motion
        if len(state['buffer']) < self.MOTION_WINDOW_FRAMES:
            return state['is_talking']

        # Get the full analysis window
        window = state['buffer']
        window_sum = sum(window)
        window_length = len(window)

        # 2. Apply Motion-Based Logic

        # Case A: Pure Static (All 0s or All 1s)
        if window_sum == 0:
            # All frames were raw '0' (closed mouth/silence)
            state['is_talking'] = 0 
        elif window_sum == window_length:
            # All frames were raw '1' (static open mouth/yawn)
            state['is_talking'] = 0 
        else:
            # Case B: Mixed States (Some 0s and some 1s)
            # This indicates dynamic switching (motion), which means active speech.
            state['is_talking'] = 1 
        
        return state['is_talking']

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
PREDICTION_THRESHOLD = 0.5 # Confidence threshold for raw model output

# --- INITIALIZE CORE COMPONENTS ---
ct = CentroidTracker() 
detector = TemporalSpeechDetector(fps=30) # Initialize the new motion detector

# Load the trained model
try:
    print(f"Loading model from: {MODEL_PATH}")
    # Suppress TensorFlow logging warnings
    tf.get_logger().setLevel('ERROR') 
    model = keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    # Exit if the model isn't available
    exit()

# --- 2. MediaPipe & OpenCV Initialization ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=5, 
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)
print("\nStarting webcam feed (Multi-Face, Motion-Stabilized)... Press 'q' to quit.")

# --- 3. Main Loop ---
while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue
        
    # Flip the image horizontally for selfie-view
    image = cv2.flip(image, 1)

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
            # Get Bounding Box from all landmarks
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

    # --- Main Logic Loop (Motion-Stabilized Prediction) ---
    for (face_ID, centroid) in tracked_faces.items():
        
        # Look up the face data using the centroid
        key_tuple = tuple(centroid)
        if key_tuple not in centroid_to_landmarks_map:
            continue
            
        face_landmarks, (x_min, y_min, x_max, y_max) = centroid_to_landmarks_map[key_tuple]
        
        # --- 1. Run Instantaneous Prediction ---
        face_crop = image[y_min:y_max, x_min:x_max]
        
        label_raw = 0 # Raw binary prediction (0 or 1)
        confidence = 0.0 # Raw probability
        
        if face_crop.size > 0:
            face_resized = cv2.resize(face_crop, (IMG_WIDTH, IMG_HEIGHT))
            face_normalized = face_resized / 255.0
            face_batch = np.expand_dims(face_normalized, axis=0)
            
            # Get raw prediction confidence
            confidence = model.predict(face_batch, verbose=0)[0][0]
            
            # Convert raw confidence to binary raw prediction (0 or 1)
            if confidence > PREDICTION_THRESHOLD:
                label_raw = 1 # Mouth is open (based on CNN)
            # else: label_raw = 0 (Mouth is closed/nearly closed)
        
        # --- 2. Apply Motion-Based Filtering ---
        # The detector checks the history buffer for 'switching'
        stabilized_state = detector.process_prediction(face_ID, label_raw)
        
        # --- 3. Update Final Label and Color ---
        if stabilized_state == 1:
            label_final = "TALKING (Motion Detected)"
            color = (0, 0, 255) # Red (B, G, R)
        else:
            label_final = "NOT TALKING (Static)"
            color = (0, 255, 0) # Green (B, G, R)

        # --- Draw on Frame ---
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)
        cv2.putText(image, f"Person {face_ID}: {label_final}", 
                    (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, color, 2)
        cv2.putText(image, f"Raw Conf: {confidence:.2f}", 
                    (x_min, y_max + 20), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, (100, 100, 100), 1)


    # --- 4. Display Image ---
    cv2.imshow('Visual Model Test (Multi-Face, Motion-Stabilized)', image)

    # --- 5. Key Press Logic ---
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- 6. Cleanup ---
print("Shutting down...")
cap.release()
cv2.destroyAllWindows()
face_mesh.close()