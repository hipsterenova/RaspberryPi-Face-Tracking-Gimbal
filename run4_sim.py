import cv2
import time
from picamera2 import Picamera2
import RPi.GPIO as GPIO

# ==========================================
# BLOCK 1: HARDWARE SETUP (DUAL MOTORS)
# ==========================================
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)

PAN_PIN = 17   # Horizontal Motor
TILT_PIN = 27  # Vertical Motor

GPIO.setup(PAN_PIN, GPIO.OUT)
GPIO.setup(TILT_PIN, GPIO.OUT)

# Start both motors at 50Hz PWM
pan_pwm = GPIO.PWM(PAN_PIN, 50)
tilt_pwm = GPIO.PWM(TILT_PIN, 50)

# Start centered (90 degrees)
current_pan_angle = 90.0
current_tilt_angle = 90.0

# Start completely relaxed (Duty Cycle 0)
pan_pwm.start(0)
tilt_pwm.start(0)

# ==========================================
# BLOCK 2: YOUR EXACT LOGIC SETTINGS
# ==========================================
THRESHOLD_X = 50        # Horizontal deadzone
THRESHOLD_Y = 40        # Vertical deadzone
STEP_DEGREE = 1.0       # Move 1 degree at a time
TIME_STEP = 0.015       # Wait 0.015 seconds between motor steps

SCREEN_CENTER_X = 320   
SCREEN_CENTER_Y = 240   

# Independent timers for each motor
last_move_time_x = time.time()
last_move_time_y = time.time()

# ==========================================
# BLOCK 3: CAMERA SETUP
# ==========================================
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (640, 480)})
picam2.configure(config)
picam2.start()

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

print("Starting Master Dual Gimbal (Sequential Logic)...")
print("Priority: Fix Horizontal FIRST, then fix Vertical.")

# ==========================================
# BLOCK 4: THE MAIN TRACKING LOOP
# ==========================================
try:
    while True:
        frame = picam2.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Draw the "Safe Zone" Target Box in the center of the screen
        cv2.rectangle(frame, 
                     (SCREEN_CENTER_X - THRESHOLD_X, SCREEN_CENTER_Y - THRESHOLD_Y), 
                     (SCREEN_CENTER_X + THRESHOLD_X, SCREEN_CENTER_Y + THRESHOLD_Y), 
                     (255, 0, 0), 2)

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))

        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            
            # Draw Face Tracking Box and Center Dot
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            face_x = x + (w // 2)
            face_y = y + (h // 2)
            cv2.circle(frame, (face_x, face_y), 5, (0, 0, 255), -1)
            
            # Calculate Distances (Errors)
            distance_x = face_x - SCREEN_CENTER_X
            distance_y = face_y - SCREEN_CENTER_Y
            
            current_time = time.time()
            
            # ==========================================
            # MASTER LOGIC: HORIZONTAL FIRST, THEN VERTICAL
            # ==========================================
            
            # STEP 1: IS HORIZONTAL OUT OF BOUNDS?
            if abs(distance_x) > THRESHOLD_X:
                # Silence the Vertical motor while Horizontal is working
                tilt_pwm.ChangeDutyCycle(0) 
                
                # Check 0.015s timer
                if (current_time - last_move_time_x) >= TIME_STEP:
                    if distance_x > 0:
                        current_pan_angle -= STEP_DEGREE
                    else:
                        current_pan_angle += STEP_DEGREE
                    
                    # Apply constraints and math
                    current_pan_angle = max(0.0, min(180.0, current_pan_angle))
                    duty_cycle_x = 2.5 + (10.0 * current_pan_angle / 180.0)
                    pan_pwm.ChangeDutyCycle(duty_cycle_x)
                    
                    last_move_time_x = current_time

            # STEP 2: HORIZONTAL IS GOOD. IS VERTICAL OUT OF BOUNDS?
            elif abs(distance_y) > THRESHOLD_Y:
                # Silence the Horizontal motor because it is perfectly centered
                pan_pwm.ChangeDutyCycle(0)
                
                # Check 0.015s timer
                if (current_time - last_move_time_y) >= TIME_STEP:
                    if distance_y > 0:
                        current_tilt_angle -= STEP_DEGREE
                    else:
                        current_tilt_angle += STEP_DEGREE
                    
                    # Apply constraints and math
                    current_tilt_angle = max(0.0, min(180.0, current_tilt_angle))
                    duty_cycle_y = 2.5 + (10.0 * current_tilt_angle / 180.0)
                    tilt_pwm.ChangeDutyCycle(duty_cycle_y)
                    
                    last_move_time_y = current_time
            
            # STEP 3: BOTH ARE PERFECTLY CENTERED
            else:
                pan_pwm.ChangeDutyCycle(0)
                tilt_pwm.ChangeDutyCycle(0)
                
        # Condition: No face detected at all
        else:
            pan_pwm.ChangeDutyCycle(0)
            tilt_pwm.ChangeDutyCycle(0)

        cv2.imshow('Master Dual Gimbal', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Safely power down everything
    pan_pwm.ChangeDutyCycle(0)
    tilt_pwm.ChangeDutyCycle(0)
    pan_pwm.stop()
    tilt_pwm.stop()
    GPIO.cleanup()
    picam2.stop()
    cv2.destroyAllWindows()
    print("System safely shut down.")