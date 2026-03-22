import cv2
import time
from picamera2 import Picamera2
import RPi.GPIO as GPIO

# ==========================================
# BLOCK 1: HARDWARE SETUP (HORIZONTAL ONLY)
# ==========================================
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)

# Using GPIO 17 for the Pan (Horizontal) Motor
PAN_PIN = 17
GPIO.setup(PAN_PIN, GPIO.OUT)

pan_pwm = GPIO.PWM(PAN_PIN, 50)
current_pan_angle = 90.0

# Start relaxed with 0 duty cycle
pan_pwm.start(0)

# ==========================================
# BLOCK 2: YOUR EXACT LOGIC SETTINGS
# ==========================================
THRESHOLD_X = 40        # Distance threshold (Deadzone) - Slightly wider for horizontal
STEP_DEGREE = 1.0       # Move 1 degree at a time
TIME_STEP = 0.2      # Wait 0.015 seconds between motor steps

SCREEN_CENTER_X = 320   # Center of the X axis (640 / 2)
last_move_time = time.time()

# ==========================================
# BLOCK 3: CAMERA SETUP
# ==========================================
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (640, 480)})
picam2.configure(config)
picam2.start()

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

print("Starting Horizontal-Only PWM Gimbal...")
print("Look for the VERTICAL blue lines!")

# ==========================================
# BLOCK 4: THE MAIN TRACKING LOOP
# ==========================================
try:
    while True:
        # Constantly grab fresh frames
        frame = picam2.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Draw the VERTICAL threshold lines so you can see the horizontal deadzone
        cv2.line(frame, (SCREEN_CENTER_X - THRESHOLD_X, 0), (SCREEN_CENTER_X - THRESHOLD_X, 480), (255, 0, 0), 1)
        cv2.line(frame, (SCREEN_CENTER_X + THRESHOLD_X, 0), (SCREEN_CENTER_X + THRESHOLD_X, 480), (255, 0, 0), 1)

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))

        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            
            # Draw box and calculate HORIZONTAL distance
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            face_x = x + (w // 2)
            face_y = y + (h // 2)
            cv2.circle(frame, (face_x, face_y), 5, (0, 0, 255), -1)
            
            # X-Axis Distance (Error)
            distance_x = face_x - SCREEN_CENTER_X
            
            # ==========================================
            # YOUR EXACT LOGIC (Adapted for X Axis)
            # ==========================================
            # Condition A: Face is OUTSIDE the left/right threshold
            if abs(distance_x) > THRESHOLD_X:
                current_time = time.time()
                
                # Check if 0.015 seconds have passed
                if (current_time - last_move_time) >= TIME_STEP:
                    
                    # NOTE: If the camera pans the WRONG way, swap the += and -= below!
                    if distance_x > 0:
                        current_pan_angle -= STEP_DEGREE  # Move Right
                    else:
                        current_pan_angle += STEP_DEGREE  # Move Left
                    
                    # Keep angle inside the physical hardware limits
                    current_pan_angle = max(0.0, min(180.0, current_pan_angle))
                    
                    # Apply YOUR exact formula
                    duty_cycle = 2.5 + (10.0 * current_pan_angle / 180.0)
                    pan_pwm.ChangeDutyCycle(duty_cycle)
                    
                    last_move_time = current_time
            
            # Condition B: Face is INSIDE the threshold (Centered horizontally)
            else:
                pan_pwm.ChangeDutyCycle(0) # Cut power, stop vibration
                
        else:
            # Condition C: No face seen
            pan_pwm.ChangeDutyCycle(0) # Cut power

        cv2.imshow('Horizontal PWM Gimbal', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Safely power down
    pan_pwm.ChangeDutyCycle(0)
    pan_pwm.stop()
    GPIO.cleanup()
    picam2.stop()
    cv2.destroyAllWindows()
    print("Horizontal Test finished.")