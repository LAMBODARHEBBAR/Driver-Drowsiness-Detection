# Import necessary libraries
import cv2  # OpenCV for image processing
import numpy as np  # Numpy for array-related functions
import dlib  # Dlib for face detection and landmark detection
from imutils import face_utils  # Utilities for face operations
import pygame  # Pygame for playing alert sound

# Initialize Pygame mixer for sound
pygame.mixer.init()
# Load alert sound
alert_sound = pygame.mixer.Sound("Alert_Sound.mp3")

# Initialize the camera
cap = cv2.VideoCapture(0)

# Check if the camera is opened successfully
if not cap.isOpened():
    print("Error: Unable to open camera.")
    exit()

# Initialize face detector and landmark detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Variables to track eye state
sleep = 0
drowsy = 0
active = 0
status = ""
color = (0, 0, 0)

# Variable to track the duration of continuous eye closure
eye_closure_timer = 0
eye_closure_threshold = 3 * 30  # Adjust this value for the desired duration (3 seconds assuming 30 frames per second)

# Function to compute distance between two points
def compute(ptA, ptB):
    dist = np.linalg.norm(ptA - ptB)
    return dist

# Function to determine if eyes are blinked
def blinked(a, b, c, d, e, f):
    # Calculate distances between various points
    up = compute(b, d) + compute(c, e)
    down = compute(a, f)
    ratio = up / (2.0 * down)

    # Classify blink based on ratio
    if ratio > 0.25:
        return 2  # Fully blinked
    elif ratio > 0.21 and ratio <= 0.25:
        return 1  # Partially blinked
    else:
        return 0  # Not blinked

# Main loop for video processing
while True:
    # Read frame from camera
    ret, frame = cap.read()  
    if not ret:
        print("Error: Unable to read frame from camera.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert frame to grayscale

    faces = detector(gray)  # Detect faces in grayscale frame

    # Iterate through detected faces
    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()

        face_frame = frame.copy()
        # Draw rectangle around the face
        cv2.rectangle(face_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Detect landmarks in the face
        landmarks = predictor(gray, face)
        landmarks = face_utils.shape_to_np(landmarks)

        # Check blinking for left and right eyes
        left_blink = blinked(landmarks[36], landmarks[37], landmarks[38], landmarks[41], landmarks[40],
                             landmarks[39])
        right_blink = blinked(landmarks[42], landmarks[43], landmarks[44], landmarks[47], landmarks[46],
                              landmarks[45])

        # Determine eye state and update status
        if left_blink == 0 or right_blink == 0:
            sleep += 1
            drowsy = 0
            active = 0
            # Increment eye closure timer
            eye_closure_timer += 1  
            # Check if eye closure duration exceeds threshold for alarm
            if sleep > 6 and eye_closure_timer > eye_closure_threshold:  
                status = "SLEEPING !!!"
                color = (255, 0, 0)
                alert_sound.play()  # Trigger the alarm
        elif left_blink == 1 or right_blink == 1:
            sleep = 0
            active = 0
            drowsy += 1
            if drowsy > 6:
                status = "Drowsy !"
                color = (0, 0, 255)
        else:
            drowsy = 0
            sleep = 0
            active += 1
            if active > 6:
                status = "Active :)"
                color = (0, 255, 0)

        # Display status on frame
        cv2.putText(frame, status, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

        # Draw landmarks on the face
        for n in range(0, 68):
            (x, y) = landmarks[n]
            cv2.circle(face_frame, (x, y), 1, (255, 255, 255), -1)

        # Display frames
        cv2.imshow("Frame", frame)
        cv2.imshow("Result of detector", face_frame)
        key = cv2.waitKey(1)
        if key == 27:  # Press 'Esc' key to exit
            break

# Release resources
cap.release()
cv2.destroyAllWindows()
