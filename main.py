import cv2
from scipy.spatial import distance
import dlib
from imutils import face_utils
import numpy as np
import csv

# Function to calculate eye aspect ratio (EAR)
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Load face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

# Indices of facial landmarks for left and right eyes
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# Open video capture
cap = cv2.VideoCapture(0)

# Initialize variables
ear_values = []
frame_count = 0

# Loop over frames from the video
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocessing: Convert to grayscale, detect faces
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    subjects = detector(gray, 0)

    # Loop over detected faces
    for subject in subjects:
        # Detect facial landmarks
        shape = predictor(gray, subject)
        shape = face_utils.shape_to_np(shape)

        # Extract left and right eye coordinates
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]

        # Calculate EAR for each eye
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        # Average EAR of both eyes
        ear = (leftEAR + rightEAR) / 2.0
        ear_values.append(ear)

        # Draw eyes on the frame
        for (x, y) in leftEye:
            cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)
        for (x, y) in rightEye:
            cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)

        # Connect landmarks with lines
        cv2.line(frame, tuple(leftEye[1]), tuple(leftEye[5]), (0, 255, 0), 1)
        cv2.line(frame, tuple(leftEye[2]), tuple(leftEye[4]), (0, 255, 0), 1)
        cv2.line(frame, tuple(leftEye[0]), tuple(leftEye[3]), (0, 255, 0), 1)
        cv2.line(frame, tuple(rightEye[1]), tuple(rightEye[5]), (0, 255, 0), 1)
        cv2.line(frame, tuple(rightEye[2]), tuple(rightEye[4]), (0, 255, 0), 1)
        cv2.line(frame, tuple(rightEye[0]), tuple(rightEye[3]), (0, 255, 0), 1)

    # Display the frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

    # Increment frame count
    frame_count += 1

    # Check if it's time to calculate EAR and save to CSV
    if frame_count == 10:  # Calculate EAR every 10 frames
        with open('ear_values.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(ear_values)
        frame_count = 0
        ear_values = []

# Release resources
cap.release()
cv2.destroyAllWindows()
