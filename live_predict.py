import tensorflow as tf
import numpy as np
import librosa
import pyaudio
import time
from pathlib import Path
import sys

# --- 1. CONFIGURATION ---
# CRITICAL: These MUST match your training parameters
SR = 16000
N_MELS = 64
FRAME_DURATION_SEC = 0.96 # Your VAD window size
N_SAMPLES = int(SR * FRAME_DURATION_SEC)
N_FFT = 400
HOP_LENGTH = 160
MODEL_PATH = r"D:\python\ml_project\vad_transfer_model.h5" 

# PyAudio Configuration
CHUNK = N_SAMPLES # Process exactly one VAD window size at a time
FORMAT = pyaudio.paInt16 # 16-bit audio format
CHANNELS = 1

# --- 2. PREPROCESSING FUNCTION ---

def preprocess_audio_chunk(audio_data):
    """Converts a raw NumPy audio chunk into a normalized spectrogram array."""
    
    # 1. Convert raw bytes to float32 NumPy array (required by Librosa)
    audio_chunk = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0 
    
    # 2. Compute Mel Spectrogram (same parameters as training)
    mel_spectrogram = librosa.feature.melspectrogram(
        y=audio_chunk, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS, fmax=SR/2
    )
    
    # 3. Convert to Logarithmic Scale (Decibels)
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
    
    # 4. Normalize and Reshape to (1, 64, 97, 1)
    min_val = -80.0
    max_val = 0.0
    spectrogram = (log_mel_spectrogram - min_val) / (max_val - min_val)
    
    # Add Channel and Batch dimensions
    spectrogram = np.expand_dims(spectrogram, axis=-1)
    spectrogram = np.expand_dims(spectrogram, axis=0)
    
    return spectrogram

# --- 3. MAIN REAL-TIME LOOP ---

def run_live_prediction():
    """Initializes mic stream and runs continuous prediction."""
    try:
        # Load the trained model
        model = tf.keras.models.load_model(MODEL_PATH)
        print(f"DEBUG: Model loaded successfully from {MODEL_PATH}")

        # Initialize PyAudio
        p = pyaudio.PyAudio()
        
        # Open Audio Stream
        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=SR,
                        input=True,
                        frames_per_buffer=CHUNK)

        print("\n--- Starting Live VAD Prediction ---")
        print(f"Listening for {FRAME_DURATION_SEC}s chunks (Press Ctrl+C to stop)...")

        label_map = {0: "NON-SPEECH 🤫", 1: "SPEECH 🗣️"}

        while True:
            # 1. Read one chunk of audio from the microphone
            audio_data = stream.read(CHUNK, exception_on_overflow=False)
            
            # 2. Preprocess the raw chunk into a spectrogram
            input_data = preprocess_audio_chunk(audio_data)

            # 3. Predict
            prediction = model.predict(input_data, verbose=0)
            
            # 4. Interpret Result
            predicted_class_index = np.argmax(prediction[0])
            confidence = prediction[0][predicted_class_index]
            predicted_label = label_map[predicted_class_index]

            # 5. Output Result
            print(f"[{time.strftime('%H:%M:%S')}] Prediction: {predicted_label} (Conf: {confidence*100:.1f}%)", end='\r')
            
    except KeyboardInterrupt:
        print("\nStopping VAD stream.")
    except Exception as e:
        print(f"\nCRITICAL ERROR during stream: {e}", file=sys.stderr)
    finally:
        # Stop and close the stream and PyAudio interface
        if 'stream' in locals() and stream.is_active():
            stream.stop_stream()
            stream.close()
        if 'p' in locals():
            p.terminate()

# --- 4. RUN SCRIPT ---
if __name__ == "__main__":
    if not Path(MODEL_PATH).exists():
        sys.exit(f"CRITICAL ERROR: Model file not found at {MODEL_PATH}. Check path.")
        
    run_live_prediction()