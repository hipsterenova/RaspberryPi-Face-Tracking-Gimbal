import cv2
import keras
import mediapipe as mp
import numpy as np
import tensorflow as tf
import os
import time
import sys
import pyaudio
import librosa
from collections import OrderedDict
from scipy.spatial import distance as dist 
from pathlib import Path

# --- 1. MODEL & CONFIGURATION PATHS (User MUST define these) ---
# NOTE: Replace these with the actual paths to your saved models
VISUAL_MODEL_PATH = "D:\\python\\ml_project\\visual_model.keras"  # Path to the lip motion detection model
AUDIO_MODEL_PATH = "D:\\python\\ml_project\\vad_transfer_model.h5"  # Path to the voice activity detection model

# --- 2. VISUAL (Lip Motion) CONFIGURATION ---
PREDICTION_THRESHOLD = 0.5  # Confidence threshold for CNN raw output (from 0 to 1)
# Fixed size for CNN input image (must match training input)
CNN_INPUT_SIZE = (64, 64) 

# --- 3. AUDIO (Voice Activity) CONFIGURATION ---
SR = 16000 # Sample Rate (must match VAD training)
N_MELS = 64
FRAME_DURATION_SEC = 0.96 # VAD window size
N_SAMPLES = int(SR * FRAME_DURATION_SEC)
N_FFT = 400
HOP_LENGTH = 160
# PyAudio Configuration
CHUNK = N_SAMPLES
FORMAT = pyaudio.paInt16
CHANNELS = 1

# --- 4. TEMPORAL SPEECH DETECTOR CLASS (From testing_model_historybuffer.py) ---
class TemporalSpeechDetector:
    """
    Implements Motion-Based Temporal Smoothing Logic for Visual Prediction.
    Logic: If frames are mixed (0, 1, 0, 1), state is TALKING (1).
    """
    def __init__(self, fps=30):
        self.FPS = fps
        self.MOTION_WINDOW_SECONDS = 0.5 
        # Ensure MOTION_WINDOW_FRAMES is an integer
        self.MOTION_WINDOW_FRAMES = int(self.FPS * self.MOTION_WINDOW_SECONDS)
        
        self.MAX_HISTORY_FRAMES = self.MOTION_WINDOW_FRAMES 
        self.person_states = {}

    def process_prediction(self, person_ID, raw_prediction_int):
        # Initialize state if new person
        if person_ID not in self.person_states:
            self.person_states[person_ID] = {
                'buffer': [],
                'is_talking': 0 # 0: Not Talking (Static/Closed), 1: Talking (Motion Detected)
            }
        
        # Update buffer
        state = self.person_states[person_ID]
        state['buffer'].append(raw_prediction_int)
        
        # Trim buffer to max history size
        if len(state['buffer']) > self.MAX_HISTORY_FRAMES:
            state['buffer'] = state['buffer'][-self.MAX_HISTORY_FRAMES:]

        # Analysis: Check for VARIABILITY (switching) in the buffer
        if len(state['buffer']) >= self.MAX_HISTORY_FRAMES:
            # Count the occurrences of '1' (open mouth)
            ones = sum(state['buffer'])
            
            zeros = len(state['buffer']) - ones
            
            # --- Condition for Visual Talking (Motion) ---
            # Motion is detected if we have at least 20% 'ones' AND at least 20% 'zeros'
            # This filters out static "open mouth" (all 1s) and static "closed mouth" (all 0s)
            
            min_mix_count = int(self.MAX_HISTORY_FRAMES * 0.20)
            
            if ones >= min_mix_count and zeros >= min_mix_count:
                state['is_talking'] = 1 # Motion detected
            else:
                state['is_talking'] = 0 # Static (closed or static open)

        return state['is_talking']

# --- 5. AUDIO PREPROCESSING FUNCTION (From live_predict.py) ---
def preprocess_audio_chunk(audio_data):
    """Converts a raw NumPy audio chunk into a normalized spectrogram array."""
    
    # 1. Convert raw bytes to float32 NumPy array (required by Librosa)
    audio_chunk = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0 
    
    # 2. Compute Mel Spectrogram (same parameters as training)
    mel_spectrogram = librosa.feature.melspectrogram(
        y=audio_chunk, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS,
        power=2.0 # Use power=2 for magnitude squared (energy)
    )
    
    # 3. Convert to dB scale (log scale)
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
    
    # 4. Normalize (Min-Max normalization) - CRITICAL STEP
    min_db = -80.0 # Assuming all values below -80dB are noise/silent floor
    max_db = 0.0
    normalized_spec = np.clip(log_mel_spectrogram, min_db, max_db)
    normalized_spec = (normalized_spec - min_db) / (max_db - min_db)
    
    # 5. Reshape for CNN input (batch_size, height, width, channels)
    # The shape should be (1, N_MELS, time_steps, 1)
    input_data = np.expand_dims(normalized_spec, axis=0) # Add batch dimension
    input_data = np.expand_dims(input_data, axis=-1)    # Add channel dimension

    return input_data

# --- 6. CORE MULTIMODAL FUNCTION ---

def run_multimodal_monitor():
    """Initializes models, streams, and runs the main loop."""
    print("Loading models...")
    try:
        # Load Visual Model
        visual_model = tf.keras.models.load_model(VISUAL_MODEL_PATH)
        print(f"Visual Model loaded from: {VISUAL_MODEL_PATH}")
    except Exception as e:
        print(f"ERROR: Could not load Visual Model from {VISUAL_MODEL_PATH}. Check path and file. {e}")
        return

    try:
        # Load Audio Model
        audio_model = tf.keras.models.load_model(AUDIO_MODEL_PATH)
        print(f"Audio Model loaded from: {AUDIO_MODEL_PATH}")
    except Exception as e:
        print(f"ERROR: Could not load Audio Model from {AUDIO_MODEL_PATH}. Check path and file. {e}")
        return

    # Initialize Mediapipe
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1, 
        min_detection_confidence=0.5, 
        min_tracking_confidence=0.5
    )
    
    # Initialize Visual Stream (Webcam)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return
        
    # Determine FPS for the Detector
    # Use a fallback value if cap.get(CAP_PROP_FPS) fails
    fps = cap.get(cv2.CAP_PROP_FPS) 
    if fps <= 1: # Common issue where FPS is reported incorrectly/as 0
        fps = 30 
    detector = TemporalSpeechDetector(fps=fps)
    
    # Initialize Audio Stream (PyAudio)
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=SR,
                    input=True,
                    frames_per_buffer=CHUNK)

    # Main Loop
    print("\nStarting Multimodal Monitoring...")
    
    try:
        while cap.isOpened():
            # ========================
            # A. VISUAL PROCESSING
            # ========================
            ret, frame = cap.read()
            if not ret:
                break
            
            # --- DEFINE FRAME DIMENSIONS HERE (FIX for 'w' and 'h' not defined error) ---
            image = cv2.flip(frame, 1)
            h, w, _ = image.shape
            
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Process with Mediapipe
            results = face_mesh.process(image_rgb)
            
            visual_motion_detected = 0 # Default: No visual motion
            face_detected = False
            
            # Initialize face bounding box coordinates outside the detection loop
            x_min, y_min, x_max, y_max = 0, 0, 0, 0
            
            if results.multi_face_landmarks:
                face_detected = True
                
                for face_ID, face_landmarks in enumerate(results.multi_face_landmarks):
                    # 1. Extract 2D Coordinates for face bounding box and lip crop
                    x_coords = [lm.x for lm in face_landmarks.landmark]
                    y_coords = [lm.y for lm in face_landmarks.landmark]
                    
                    # Define a tight bounding box around the whole face
                    x_min, x_max = int(min(x_coords) * w), int(max(x_coords) * w)
                    y_min, y_max = int(min(y_coords) * h), int(max(y_coords) * h)
                    
                    # Add padding to the face bounding box for better visualization
                    padding = 10
                    x_min = max(0, x_min - padding)
                    y_min = max(0, y_min - padding)
                    x_max = min(w, x_max + padding)
                    y_max = min(h, y_max + padding)
                    
                    
                    # Define a crop box for the lips (adjusting based on full face box)
                    # Use a smaller region around the mouth for the CNN input
                    
                    # Calculate center coordinates
                    center_x = (x_min + x_max) // 2
                    center_y = (y_min + y_max) // 2
                    
                    # Define a square crop centered around the mouth/nose area (approx upper half of the detected face)
                    crop_size = 120 # Adjust this based on expected lip size
                    
                    crop_x_start = max(0, center_x - crop_size // 2)
                    crop_x_end = min(w, center_x + crop_size // 2)
                    
                    # Adjust y-center to focus slightly lower than the face center (on the mouth)
                    # Let's target the crop roughly around the mouth region (2/3rds down from top)
                    mouth_y_target = y_min + int((y_max - y_min) * 0.6)
                    crop_y_start = max(0, mouth_y_target - crop_size // 2)
                    crop_y_end = min(h, mouth_y_target + crop_size // 2)


                    # Crop the region of interest (mostly lips/mouth area)
                    cropped_image = image[crop_y_start:crop_y_end, crop_x_start:crop_x_end]
                    
                    if cropped_image.size != 0 and cropped_image.shape[0] > 10 and cropped_image.shape[1] > 10:
                        # 2. Preprocess for Visual CNN
                        # Resize to required input size (e.g., 64x64)
                        resized_crop = cv2.resize(cropped_image, CNN_INPUT_SIZE)
                        
                        # Normalize and reshape (1, 64, 64, 3)
                        input_tensor = resized_crop.astype('float32') / 255.0
                        input_tensor = np.expand_dims(input_tensor, axis=0)
                        
                        # 3. Predict Visual Raw State (Lip Openness)
                        prediction = visual_model.predict(input_tensor, verbose=0)
                        confidence = prediction[0][0]
                        
                        # Raw prediction: 1 (Open) if confidence > threshold, else 0 (Closed)
                        label_raw = 1 if confidence > PREDICTION_THRESHOLD else 0
                        
                        # 4. Apply Temporal Smoothing (Motion Filter)
                        # This determines if the visual state is actively 'talking' (switching)
                        visual_motion_detected = detector.process_prediction(face_ID, label_raw)
                    
            
            # ========================
            # B. AUDIO PROCESSING
            # ========================
            audio_data = stream.read(CHUNK, exception_on_overflow=False)
            input_data = preprocess_audio_chunk(audio_data)
            
            # Predict Voice Activity
            audio_prediction = audio_model.predict(input_data, verbose=0)
            
            # Output classes: 0: NON-SPEECH, 1: SPEECH
            predicted_class_index = np.argmax(audio_prediction[0])
            audio_confidence = audio_prediction[0][predicted_class_index]
            
            # Audio VAD: 1 if speech is detected, 0 otherwise
            # Add a confidence threshold for audio VAD as well (e.g., must be > 0.7)
            AUDIO_CONFIDENCE_THRESHOLD = 0.7 
            audio_speech_detected = 1 if (predicted_class_index == 1 and audio_confidence > AUDIO_CONFIDENCE_THRESHOLD) else 0

            
            # ========================
            # C. MULTIMODAL FUSION & OUTPUT LOGIC
            # ========================
            final_status_text = ""
            final_box_color = (0, 255, 0) # Default: Green (B, G, R) - Not Talking

            # --- Multimodal Decision Flow ---
            if not face_detected:
                final_status_text = "STATUS: NO FACE DETECTED"
                final_box_color = (128, 128, 128) # Grey
            elif visual_motion_detected == 0:
                # Requirement 1: Visual model active, but no lip motion detected
                final_status_text = "STATUS: NO LIP MOTION DETECTED"
                final_box_color = (0, 255, 0) # Green (Stagnant/Silence)
            elif visual_motion_detected == 1:
                # Requirement 2: Lip motion detected, check audio VAD
                if audio_speech_detected == 1:
                    # Condition: Visual Motion AND Audio Speech -> Speaking
                    final_status_text = "STATUS: PERSON IS SPEAKING"
                    final_box_color = (0, 0, 255) # Red (Active Speech)
                else:
                    # Condition: Visual Motion BUT NO Audio Speech -> Silent movement (Whispering, silent counting, VAD missed it)
                    final_status_text = "STATUS: SOUND THRESHOLD NOT REACHED"
                    final_box_color = (0, 165, 255) # Orange (Visual but no Sound)

            # --- D. DRAW OUTPUT ON FRAME ---
            
            # Draw the main status box (uses W and H which are always defined)
            cv2.rectangle(image, (20, 20), (w - 20, 80), final_box_color, -1)
            cv2.putText(image, final_status_text, 
                        (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                        1.0, (255, 255, 255), 2)
                        
            # Draw person status box if face is detected (uses X/Y coordinates)
            if face_detected:
                 # Draw face bounding box
                cv2.rectangle(image, (x_min, y_min), (x_max, y_max), final_box_color, 2)
                
                # Draw Visual Status
                vis_text = "VISUAL: Motion" if visual_motion_detected == 1 else "VISUAL: Static"
                cv2.putText(image, vis_text, (x_min, y_min - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, final_box_color, 2)
                
                # Draw Audio Status
                aud_text = f"AUDIO: Speech ({audio_confidence*100:.1f}%)" if audio_speech_detected == 1 else f"AUDIO: Non-Speech ({audio_confidence*100:.1f}%)"
                cv2.putText(image, aud_text, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, final_box_color, 2)

            # Display the frame
            cv2.imshow('Multimodal Silence Monitor', image)
            
            # Break loop on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\nStopping monitor.")
    except Exception as e:
        print(f"\nCRITICAL ERROR in main loop: {e}", file=sys.stderr)
        
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        stream.stop_stream()
        stream.close()
        p.terminate()
        face_mesh.close()

if __name__ == '__main__':
    # NOTE: Since you provided the paths in your last run, I'm using them directly here.
    # The printed message is now just a confirmation.
    print("---------------------------------------------------------")
    print("Confirmed Model Paths are set:")
    print(f"Visual: {VISUAL_MODEL_PATH}")
    print(f"Audio: {AUDIO_MODEL_PATH}")
    print("---------------------------------------------------------")
    run_multimodal_monitor()