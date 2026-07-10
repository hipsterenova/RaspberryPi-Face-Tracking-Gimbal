import cv2
import mediapipe as mp
import time

# --- 1. Initialization ---

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=2,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)

print("Starting webcam feed... Press 'q' to quit.")

# --- 2. Main Loop (as in your plan) ---
while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # To improve performance, mark the image as not writeable
    image.flags.writeable = False 
    
    # Convert the BGR image (from OpenCV) to RGB (for MediaPipe)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # --- 3. Run Analysis ---
    results = face_mesh.process(image)

    # Convert the image back to BGR for drawing
    image.flags.writeable = True 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # --- 4. Per-Face Logic ---
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            
            # Draw the FaceMesh tessellation
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION, 
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                .get_default_face_mesh_tesselation_style()) # <-- FINAL FIX

            # Draw outlines for lips, eyes, etc.
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                .get_default_face_mesh_contours_style())

    # Flip the image horizontally for a "selfie" view
    image = cv2.flip(image, 1)

    # Display the final image
    cv2.imshow('AI-Powered Library Silence Monitor', image)

    # Check for 'q' key to quit
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# --- 5. Cleanup ---
print("Shutting down...")
cap.release()
cv2.destroyAllWindows()
face_mesh.close()
