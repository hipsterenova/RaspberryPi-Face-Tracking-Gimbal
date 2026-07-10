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

# --- 1. MODEL & CONFIGURATION PATHS ---
VISUAL_MODEL_PATH = "D:\\python\\ml_project\\visual_model.keras"  
AUDIO_MODEL_PATH = r"D:\python\ml_project\vad_transfer_model.h5"

# --- 2. VISUAL CONFIGURATION ---
PREDICTION_THRESHOLD = 0.8  # Keep high threshold
CNN_INPUT_SIZE = (64, 64) 
PROCESS_WIDTH = 640
PROCESS_HEIGHT = 480
FRAME_SKIP = 2 

# --- 3. AUDIO CONFIGURATION ---
SR = 16000 
N_MELS = 64
FRAME_DURATION_SEC = 0.96 
N_SAMPLES = int(SR * FRAME_DURATION_SEC)
N_FFT = 400
HOP_LENGTH = 160
CHUNK = N_SAMPLES
FORMAT = pyaudio.paInt16
CHANNELS = 1

# --- 4. TEMPORAL SPEECH DETECTOR CLASS ---
class TemporalSpeechDetector:
    def __init__(self, fps=30):
        self.FPS = fps
        self.MOTION_WINDOW_SECONDS = 0.5 
        self.MOTION_WINDOW_FRAMES = int(self.FPS * self.MOTION_WINDOW_SECONDS)
        self.MAX_HISTORY_FRAMES = self.MOTION_WINDOW_FRAMES 
        self.person_states = {}

    def process_prediction(self, person_ID, raw_prediction_int):
        if person_ID not in self.person_states:
            self.person_states[person_ID] = {
                'buffer': [],
                'is_talking': 0 
            }
        
        state = self.person_states[person_ID]
        state['buffer'].append(raw_prediction_int)
        
        if len(state['buffer']) > self.MAX_HISTORY_FRAMES:
            state['buffer'] = state['buffer'][-self.MAX_HISTORY_FRAMES:]

        if len(state['buffer']) >= 5:
            ones = sum(state['buffer'])
            total = len(state['buffer'])
            
            if ones == 0:
                state['is_talking'] = 0 
            elif ones == total:
                state['is_talking'] = 0 
            else:
                state['is_talking'] = 1 

        return state['is_talking'], state['buffer']

# --- 5. AUDIO PREPROCESSING FUNCTION ---
def preprocess_audio_chunk(audio_data):
    try:
        audio_chunk = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0 
        mel_spectrogram = librosa.feature.melspectrogram(
            y=audio_chunk, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS,power=2.0
        )
        log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
        min_db = -80.0 
        max_db = 0.0
        normalized_spec = np.clip(log_mel_spectrogram, min_db, max_db)
        normalized_spec = (normalized_spec - min_db) / (max_db - min_db)
        input_data = np.expand_dims(normalized_spec, axis=0) 
        input_data = np.expand_dims(input_data, axis=-1)    
        return input_data
    except Exception as e:
        print(f"Audio Preprocessing Error: {e}")
        return np.zeros((1, N_MELS, 94, 1)) 

# --- 6. CORE MULTIMODAL FUNCTION ---
def run_multimodal_monitor():
    print("Loading models...")
    try:
        visual_model = tf.keras.models.load_model(VISUAL_MODEL_PATH)
        print(f"Visual Model loaded from: {VISUAL_MODEL_PATH}")
    except Exception as e:
        print(f"ERROR: Visual Model load failed. {e}")
        return

    try:
        audio_model = tf.keras.models.load_model(AUDIO_MODEL_PATH)
        print(f"Audio Model loaded from: {AUDIO_MODEL_PATH}")
    except Exception as e:
        print(f"ERROR: Audio Model load failed. {e}")
        return

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1, 
        min_detection_confidence=0.5, 
        min_tracking_confidence=0.5
    )
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, PROCESS_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, PROCESS_HEIGHT)
        
    fps = cap.get(cv2.CAP_PROP_FPS) 
    if fps <= 1: fps = 30 
    
    detector_fps = fps / FRAME_SKIP
    detector = TemporalSpeechDetector(fps=detector_fps)
    
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=SR, input=True, frames_per_buffer=CHUNK)

    print("\nStarting Multimodal Monitoring...")
    
    frame_count = 0
    last_visual_motion = 0
    last_face_detected = False
    last_bounding_box = (0, 0, 0, 0)
    
    # Debug variables
    debug_label_raw = 0
    debug_visual_conf = 0.0
    debug_buffer_str = ""
    
    # --- AUDIO SYNC VARIABLES (The Fix) ---
    audio_speech_detected_instant = 0
    audio_speech_hold_timer = 0
    AUDIO_HOLD_DURATION = 1.5 # Keep audio 'active' for 1.5 seconds after detection
    last_audio_conf = 0.0

    try:
        while cap.isOpened():
            current_time = time.time()
            
            ret, frame = cap.read()
            if not ret: break
            
            image = cv2.flip(frame, 1)
            h, w, _ = image.shape
            
            # --- VISUAL PROCESSING ---
            if frame_count % FRAME_SKIP == 0:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(image_rgb)
                
                visual_motion_detected = 0 
                face_detected = False
                
                if results.multi_face_landmarks:
                    face_detected = True
                    
                    for face_ID, face_landmarks in enumerate(results.multi_face_landmarks):
                        x_coords = [lm.x for lm in face_landmarks.landmark]
                        y_coords = [lm.y for lm in face_landmarks.landmark]
                        
                        x_min, x_max = int(min(x_coords) * w), int(max(x_coords) * w)
                        y_min, y_max = int(min(y_coords) * h), int(max(y_coords) * h)
                        
                        padding = 10
                        x_min = max(0, x_min - padding)
                        y_min = max(0, y_min - padding)
                        x_max = min(w, x_max + padding)
                        y_max = min(h, y_max + padding)
                        
                        last_bounding_box = (x_min, y_min, x_max, y_max)
                        
                        center_x = (x_min + x_max) // 2
                        crop_size = 120 
                        crop_x_start = max(0, center_x - crop_size // 2)
                        crop_x_end = min(w, center_x + crop_size // 2)
                        mouth_y_target = y_min + int((y_max - y_min) * 0.6)
                        crop_y_start = max(0, mouth_y_target - crop_size // 2)
                        crop_y_end = min(h, mouth_y_target + crop_size // 2)

                        cropped_image = image[crop_y_start:crop_y_end, crop_x_start:crop_x_end]
                        
                        if cropped_image.size != 0 and cropped_image.shape[0] > 10 and cropped_image.shape[1] > 10:
                            resized_crop = cv2.resize(cropped_image, CNN_INPUT_SIZE)
                            input_tensor = resized_crop.astype('float32') / 255.0
                            input_tensor = np.expand_dims(input_tensor, axis=0)
                            
                            prediction = visual_model.predict(input_tensor, verbose=0)
                            debug_visual_conf = prediction[0][0]
                            
                            debug_label_raw = 1 if debug_visual_conf > PREDICTION_THRESHOLD else 0
                            
                            visual_motion_detected, buffer = detector.process_prediction(face_ID, debug_label_raw)
                            debug_buffer_str = "".join(map(str, buffer[-10:]))
                            
                            print(f"Vis: {debug_visual_conf:.2f} | Aud: {last_audio_conf:.2f} | Buf: {debug_buffer_str}", end='\r')
                
                last_visual_motion = visual_motion_detected
                last_face_detected = face_detected
            else:
                visual_motion_detected = last_visual_motion
                face_detected = last_face_detected
                x_min, y_min, x_max, y_max = last_bounding_box

            frame_count += 1
            
            # --- AUDIO PROCESSING (With Memory Logic) ---
            try:
                if stream.get_read_available() >= CHUNK:
                    audio_data = stream.read(CHUNK, exception_on_overflow=False)
                    input_data = preprocess_audio_chunk(audio_data)
                    audio_prediction = audio_model.predict(input_data, verbose=0)
                    
                    predicted_class_index = np.argmax(audio_prediction[0])
                    last_audio_conf = audio_prediction[0][predicted_class_index]
                    
                    AUDIO_CONFIDENCE_THRESHOLD = 0.7 
                    
                    # Instant detection for this chunk
                    if predicted_class_index == 1 and last_audio_conf > AUDIO_CONFIDENCE_THRESHOLD:
                        audio_speech_detected_instant = 1
                        # RESET the hold timer - Speech is active right now!
                        audio_speech_hold_timer = current_time + AUDIO_HOLD_DURATION 
                    else:
                        audio_speech_detected_instant = 0
            except IOError:
                pass

            # --- AUDIO SYNC CHECK ---
            # Check if we are currently within the "hold" window
            if current_time < audio_speech_hold_timer:
                final_audio_status = 1 # Speech "Active" (Real or Memory)
            else:
                final_audio_status = 0 # Silence

            # --- OUTPUT LOGIC ---
            final_status_text = ""
            final_box_color = (0, 255, 0) 

            if not face_detected:
                final_status_text = "STATUS: NO FACE DETECTED"
                final_box_color = (128, 128, 128)
            elif visual_motion_detected == 0:
                final_status_text = "STATUS: NO LIP MOTION DETECTED"
                final_box_color = (0, 255, 0) 
            elif visual_motion_detected == 1:
                # Use the SYNCED audio status, not the instant one
                if final_audio_status == 1:
                    final_status_text = "STATUS: PERSON IS SPEAKING"
                    final_box_color = (0, 0, 255) 
                else:
                    final_status_text = "STATUS: SOUND THRESHOLD NOT REACHED"
                    final_box_color = (0, 165, 255) 

            cv2.rectangle(image, (20, 20), (w - 20, 80), final_box_color, -1)
            cv2.putText(image, final_status_text, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            
            # Debug Info
            cv2.putText(image, f"Vis Conf: {debug_visual_conf:.2f}", (20, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(image, f"Aud Conf: {last_audio_conf:.2f} (Hold: {final_audio_status})", (20, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(image, f"Buffer: [{debug_buffer_str}]", (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            if face_detected:
                cv2.rectangle(image, (x_min, y_min), (x_max, y_max), final_box_color, 2)
                vis_text = "VISUAL: Motion" if visual_motion_detected == 1 else "VISUAL: Static"
                cv2.putText(image, vis_text, (x_min, y_min - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, final_box_color, 2)
                
                # Show hold status
                aud_text_base = f"AUDIO: Speech" if final_audio_status == 1 else f"AUDIO: Non-Speech"
                aud_text = f"{aud_text_base} ({last_audio_conf*100:.0f}%)"
                cv2.putText(image, aud_text, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, final_box_color, 2)

            cv2.imshow('Multimodal Silence Monitor', image)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

    except KeyboardInterrupt:
        print("\nStopping monitor.")
    except Exception as e:
        print(f"\nCRITICAL ERROR: {e}", file=sys.stderr)
    finally:
        cap.release()
        cv2.destroyAllWindows()
        stream.stop_stream()
        stream.close()
        p.terminate()
        face_mesh.close()

if __name__ == '__main__':
    run_multimodal_monitor()