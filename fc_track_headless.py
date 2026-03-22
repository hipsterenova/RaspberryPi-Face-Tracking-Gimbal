import cv2
import time
from picamera2 import Picamera2
import RPi.GPIO as GPIO

# ==========================================
# BLOCK 1: HARDWARE SETUP (DUAL MOTORS)
# ==========================================
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)

PAN_PIN = 17   
TILT_PIN = 27  

GPIO.setup(PAN_PIN, GPIO.OUT)
GPIO.setup(TILT_PIN, GPIO.OUT)

pan_pwm = GPIO.PWM(PAN_PIN, 50)
tilt_pwm = GPIO.PWM(TILT_PIN, 50)

# Start completely centered and relaxed
HOME_ANGLE = 90.0
current_pan_angle = HOME_ANGLE
current_tilt_angle = HOME_ANGLE

pan_pwm.start(0)
tilt_pwm.start(0)

# ==========================================
# BLOCK 2: YOUR EXACT LOGIC SETTINGS
# ==========================================
HEADLESS = True         # Set to True to run without a monitor/GUI over SSH

THRESHOLD_X = 50        
THRESHOLD_Y = 40        
STEP_DEGREE = 1.0       
TIME_STEP = 0.015       

SCREEN_CENTER_X = 320   
SCREEN_CENTER_Y = 240   

# --- NEW: SCANNING / RESET SETTINGS ---
RESET_DELAY = 3.0       # Wait 3 seconds after losing the face before going home

# Independent timers
last_move_time_x = time.time()
last_move_time_y = time.time()
last_face_time = time.time() # Tracks the exact moment we last saw a face

# ==========================================
# BLOCK 3: CAMERA SETUP
# ==========================================
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (640, 480)})
picam2.configure(config)
picam2.start()

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

print("Starting Master Dual Gimbal...")
print("Option 1 Active: Will smoothly Return to Home (90, 90) if no face is seen for 3 seconds.")

# ==========================================
# BLOCK 4: THE MAIN TRACKING LOOP
# ==========================================
try:
    while True:
        frame = picam2.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        current_time = time.time()
        
        # Draw the "Safe Zone" Box (We keep drawing just in case we want to save video later)
        cv2.rectangle(frame, 
                     (SCREEN_CENTER_X - THRESHOLD_X, SCREEN_CENTER_Y - THRESHOLD_Y), 
                     (SCREEN_CENTER_X + THRESHOLD_X, SCREEN_CENTER_Y + THRESHOLD_Y), 
                     (255, 0, 0), 2)

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))

        # ==========================================
        # SCENARIO A: FACE IS DETECTED!
        # ==========================================
        if len(faces) > 0:
            # Update the timer because we see a face right now!
            last_face_time = current_time 
            
            (x, y, w, h) = faces[0]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            face_x = x + (w // 2)
            face_y = y + (h // 2)
            cv2.circle(frame, (face_x, face_y), 5, (0, 0, 255), -1)
            
            distance_x = face_x - SCREEN_CENTER_X
            distance_y = face_y - SCREEN_CENTER_Y
            
            # --- TRACKING LOGIC (Horizontal First, then Vertical) ---
            if abs(distance_x) > THRESHOLD_X:
                tilt_pwm.ChangeDutyCycle(0) 
                if (current_time - last_move_time_x) >= TIME_STEP:
                    if distance_x > 0:
                        current_pan_angle -= STEP_DEGREE
                    else:
                        current_pan_angle += STEP_DEGREE
                    
                    current_pan_angle = max(0.0, min(180.0, current_pan_angle))
                    duty_cycle_x = 2.5 + (10.0 * current_pan_angle / 180.0)
                    pan_pwm.ChangeDutyCycle(duty_cycle_x)
                    last_move_time_x = current_time

            elif abs(distance_y) > THRESHOLD_Y:
                pan_pwm.ChangeDutyCycle(0)
                if (current_time - last_move_time_y) >= TIME_STEP:
                    if distance_y > 0:
                        current_tilt_angle -= STEP_DEGREE
                    else:
                        current_tilt_angle += STEP_DEGREE
                    
                    current_tilt_angle = max(0.0, min(180.0, current_tilt_angle))
                    duty_cycle_y = 2.5 + (10.0 * current_tilt_angle / 180.0)
                    tilt_pwm.ChangeDutyCycle(duty_cycle_y)
                    last_move_time_y = current_time
            else:
                pan_pwm.ChangeDutyCycle(0)
                tilt_pwm.ChangeDutyCycle(0)
                
        # ==========================================
        # SCENARIO B: NO FACE DETECTED
        # ==========================================
        else:
            time_since_last_face = current_time - last_face_time
            
            # 1. Has it been more than 3 seconds? (Time to go home!)
            if time_since_last_face > RESET_DELAY:
                cv2.putText(frame, "RETURNING TO HOME...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Smoothly step the Horizontal motor back to 90.0
                if abs(current_pan_angle - HOME_ANGLE) > 0.1:
                    tilt_pwm.ChangeDutyCycle(0)
                    if (current_time - last_move_time_x) >= TIME_STEP:
                        if current_pan_angle > HOME_ANGLE:
                            current_pan_angle = max(HOME_ANGLE, current_pan_angle - STEP_DEGREE)
                        else:
                            current_pan_angle = min(HOME_ANGLE, current_pan_angle + STEP_DEGREE)
                        pan_pwm.ChangeDutyCycle(2.5 + (10.0 * current_pan_angle / 180.0))
                        last_move_time_x = current_time
                
                # Smoothly step the Vertical motor back to 90.0
                elif abs(current_tilt_angle - HOME_ANGLE) > 0.1:
                    pan_pwm.ChangeDutyCycle(0)
                    if (current_time - last_move_time_y) >= TIME_STEP:
                        if current_tilt_angle > HOME_ANGLE:
                            current_tilt_angle = max(HOME_ANGLE, current_tilt_angle - STEP_DEGREE)
                        else:
                            current_tilt_angle = min(HOME_ANGLE, current_tilt_angle + STEP_DEGREE)
                        tilt_pwm.ChangeDutyCycle(2.5 + (10.0 * current_tilt_angle / 180.0))
                        last_move_time_y = current_time
                
                # We are officially Home! Cut power.
                else:
                    cv2.putText(frame, "STANDING BY", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    pan_pwm.ChangeDutyCycle(0)
                    tilt_pwm.ChangeDutyCycle(0)
            
            # 2. It hasn't been 3 seconds yet. Just wait patiently and keep quiet.
            else:
                pan_pwm.ChangeDutyCycle(0)
                tilt_pwm.ChangeDutyCycle(0)

        # --- HEADLESS TOGGLE ---
        if not HEADLESS:
            cv2.imshow('Master Dual Gimbal', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

except KeyboardInterrupt:
    print("\nScript stopped manually by user (Ctrl+C).")

finally:
    pan_pwm.ChangeDutyCycle(0)
    tilt_pwm.ChangeDutyCycle(0)
    pan_pwm.stop()
    tilt_pwm.stop()
    GPIO.cleanup()
    picam2.stop()
    cv2.destroyAllWindows()
    print("System safely shut down.")
