# Autonomous Dual-Axis Face-Tracking Gimbal

### Project Overview
A real-time, dual-axis autonomous face-tracking camera gimbal engineered using a Raspberry Pi, OpenCV, and direct PWM motor control. The system continuously processes a live video feed, calculates spatial error coordinates, and mechanically drives two SG90 servo motors to keep a target dynamically locked in the center of the frame.

*(Tip: Upload a GIF or photo of your gimbal moving on your desk and put it right here!)*

---

### 🛠️ Technical Stack & Specifications
* **Hardware:** Raspberry Pi, Picamera2 module, 2x TowerPro SG90 micro servos (Pan/Tilt).
* **Software:** Python 3, OpenCV (Haar Cascades for Object Detection), `RPi.GPIO` library.
* **Core Metrics:**
  * **Video Processing:** 640x480 resolution at real-time framerates.
  * **Motor Control:** 50Hz base frequency, utilizing a strict 2.5% to 12.5% duty cycle range to map 0° to 180° of physical rotation.
  * **Actuation Resolution:** Constrained to 1.0-degree micro-steps executing every 15 milliseconds (0.015s) for smooth kinetic movement.
  * **Deadzone Bounding Box:** A 100x80 pixel safe zone (±50px X-axis, ±40px Y-axis) to prevent micro-stuttering.

---

### 🚀 Key Engineering Achievements
* **Custom Proportional Control Algorithm:** Bypassed standard high-level motor libraries to write the raw mathematical translation from pixel-error to physical servo angles: `Duty Cycle = 2.5 + (10.0 * Target_Angle / 180.0)`.
* **Sequential Priority Logic:** Designed a state machine to prevent power spikes and mechanical fighting. The system corrects the X-axis error completely before allocating power to correct the Y-axis error.
* **"Search and Rescue" State Machine:** Implemented a 3.0-second fallback timer. If the computer vision model loses the target (or chases a false positive), the system pauses, cuts power, and smoothly interpolates both axes back to the absolute 90.0° center to await target reacquisition.
* **Headless Deployment:** Built a headless toggle to bypass GUI rendering (`cv2.imshow`), significantly reducing CPU overhead when deploying the script via SSH.

---

### 🧠 Technical Challenges Overcome

**1. The RTOS Jitter Problem (Hardware vs. Software PWM)**
* **The Issue:** The motors violently vibrated in place even when the face was perfectly centered. 
* **The Root Cause:** Standard Raspberry Pi OS uses preemptive multitasking, not a Real-Time Operating System (RTOS). The CPU slightly overslept its software-based PWM timings (e.g., waking up at 1.55ms instead of 1.50ms). To a servo, a 0.05ms drift is interpreted as a command to move several degrees, causing violent jitter.
* **The Fix:** Engineered a "power-cut" logic loop. The microsecond the facial coordinates enter the pixel deadzone, the software explicitly sends a `ChangeDutyCycle(0)` command, dropping the electrical signal to 0V. This physically relaxes the motor and instantly eliminated 100% of the jitter.

**2. Control Loop Synchronization ("Flying Blind")**
* **The Issue:** The camera was wildly overshooting the target and spinning past the subject.
* **The Root Cause:** Asynchronous timing loops. I initially restricted the heavy CV calculations to run every 0.5 seconds to save CPU, but the motor control loop was executing every 0.015 seconds. The motors were taking 33 blind physical steps based on stale coordinate data. 
* **The Fix:** Synchronized the control loop. I forced the spatial error calculation to run on every single frame sequentially, ensuring the motor only stepped if the *most recent* frame confirmed the target was out of bounds.

**3. Hardware Allocation Conflicts**
* **The Issue:** Encountered fatal `lgpio.error: 'GPIO not allocated'` exceptions during rapid testing.
* **The Root Cause:** Improper shutdown handling leaving the GPIO pins locked in the Linux kernel memory by crashed Python scripts or background daemons. 
* **The Fix:** Implemented standard `try/except/finally` blocks to guarantee `GPIO.cleanup()` execution on `Ctrl+C` interrupts, allowing for clean boot-ups every time.

---

### 💻 How to Run
1. Clone this repository to your Raspberry Pi.
2. Ensure you have the required libraries installed: `pip install opencv-python RPi.GPIO`
3. Connect your Pan servo to `GPIO 17` and Tilt servo to `GPIO 27`.
4. Run the script:
   ```bash
   python master_gimbal.py
