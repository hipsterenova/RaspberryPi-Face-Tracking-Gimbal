from gpiozero import Servo
from time import sleep

# Initialize the vertical motor
tilt_servo = Servo(27, min_pulse_width=0.0005, max_pulse_width=0.0025)

def move_and_relax(target_position):
    # 1. Send the signal to move
    tilt_servo.value = target_position
    
    # 2. Give the motor 0.5 seconds to physically travel there
    sleep(0.5) 
    
    # 3. Cut the signal so it stops vibrating!
    tilt_servo.detach()

print("Starting Jitter-Free Vertical Test...")

try:
    print("1. Moving to Center (0.0)")
    move_and_relax(0.0)
    sleep(1.5) # Wait in silence
    
    print("2. Tilting slightly one way (-0.3)")
    move_and_relax(-0.3)
    sleep(1.5)
    
    print("3. Moving back to Center (0.0)")
    move_and_relax(0.0)
    sleep(1.5)
    
    print("4. Tilting slightly the other way (0.3)")
    move_and_relax(0.3)
    sleep(1.5)
    
    print("5. Final Return to Center (0.0)")
    move_and_relax(0.0)
    
    print("Vertical test complete!")

except KeyboardInterrupt:
    print("\nTest stopped early.")