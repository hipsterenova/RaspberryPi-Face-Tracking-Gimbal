import cv2
import mediapipe as mp
import time
import math # We need this for distance calculation

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
print("Look for the 'Lip Ratio' value on your screen.")

# --- 2. Main Loop (as in your plan) ---
while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Get image dimensions
    image_height, image_width, _ = image.shape

    # To improve performance, mark the image as not writeable
    image.flags.writeable = False 
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # --- 3. Run Analysis ---
    results = face_mesh.process(image)

    # Convert the image back to BGR for drawing
    image.flags.writeable = True 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    lip_ratio = 0.0 # Default value

    # --- 4. Per-Face Logic ---
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            
            # --- START of new code ---

            # Get the list of all 478 landmarks
            landmarks = face_landmarks.landmark
            
            # -- Get coordinates for our key points --
            # Note: .x and .y are NORMALIZED (0.0 to 1.0)
            
            # LIP GAP points
            lip_top_inner = landmarks[13]
            lip_bottom_inner = landmarks[14]
            
            # MOUTH WIDTH points (corners)
            mouth_left_corner = landmarks[78]
            mouth_right_corner = landmarks[308]

            # -- Calculate vertical lip gap --
            # We just need the 'y' coordinate difference
            lip_gap_y = abs(lip_top_inner.y - lip_bottom_inner.y)
            
            # -- Calculate horizontal mouth width --
            # We just need the 'x' coordinate difference
            mouth_width_x = abs(mouth_right_corner.x - mouth_left_corner.x)

            # -- Calculate the Ratio --
            # Add a small value (epsilon) to avoid dividing by zero
            if mouth_width_x > 0.0001: 
                lip_ratio = lip_gap_y / mouth_width_x
            
            # --- END of new code ---


            # --- Draw the CONTOURS (lips, eyes, etc.) ---
            # We removed the Tesselation drawing
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                .get_default_face_mesh_contours_style())

        # --- 5. Display Value on Screen ---
    # We do this *outside* the face loop

    # --- THIS IS THE FIX ---
    # Flip the image *first* for a "selfie" view
    image = cv2.flip(image, 1)
    # --- END OF FIX ---
        
    # Format the ratio to 4 decimal places
    display_text = f"Lip Ratio: {lip_ratio:.4f}"
        
    # Now, put the text on the *already-flipped* image
    cv2.putText(image, 
                display_text, 
                (30, 60), # Position (top-left corner)
                cv2.FONT_HERSHEY_SIMPLEX, 
                1, # Font scale
                (0, 255, 0), # Color (Green)
                2) # Thickness

    # Display the final image
    cv2.imshow('AI-Powered Library Silence Monitor', image)

        # Also print to console, (optional)
        # print(f"Lip Ratio: {lip_ratio:.4f}") 

        # Check for 'q' key to quit
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# --- 6. Cleanup ---
print("Shutting down...")
cap.release()
cv2.destroyAllWindows()
face_mesh.close()