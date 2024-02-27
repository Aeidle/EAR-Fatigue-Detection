import cv2
import dlib
import numpy as np
import os
from scipy.spatial import distance
from imutils import face_utils
import pygame
from threading import Thread
from datetime import datetime
import csv

# Initialize pygame mixer for playing audio and Load the alert sound
pygame.mixer.init()
alert_sound = pygame.mixer.Sound("audio/wake_up.wav")

# Function to calculate eye aspect ratio (EAR)
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Function to save EAR values with timestamps to CSV
def save_ear_values(ear_values):
    file_exists = os.path.isfile('output/ear_values.csv')
    with open('output/ear_values.csv', mode='a', newline='\n') as file:
        writer = csv.writer(file)
        if not file_exists or os.stat('output/ear_values.csv').st_size == 0:
            writer.writerow(['time', 'EAR'])
        for ear in ear_values:
            timestamp = datetime.now().strftime("%M:%S:%f")
            writer.writerow([f"{timestamp}", f"{ear}"])

# Initialize variables
ear_values = []
frame_count = 0
alert_count = 0

# Load face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

# Indices of facial landmarks for left and right eyes
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]


# Function to process frames
def process_frame(frame, frame_count, output_dir):
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Noise reduction using bilateral filter
    blurred = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # Histogram equalization for enhancing contrast
    equalized = cv2.equalizeHist(gray)
    
    # Edge detection using Sobel
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
    edges = cv2.sqrt(cv2.addWeighted(cv2.pow(sobelx, 2), 1.0, cv2.pow(sobely, 2), 1.0, 0))
    edges = cv2.normalize(edges, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    
    # Morphology techniques (Closing operation to close gaps between edges)
    kernel = np.ones((5,5),np.uint8)
    closing = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    
    # Improve contrast using OpenCV convertScaleAbs
    contrasted = cv2.convertScaleAbs(gray, alpha=1.5, beta=0)
    
    # Save processed frames into respective directories
    cv2.imwrite(os.path.join(output_dir, 'original', f'frame-{frame_count}.jpg'), frame)
    cv2.imwrite(os.path.join(output_dir, 'sobel', f'frame-{frame_count}.jpg'), edges)
    cv2.imwrite(os.path.join(output_dir, 'grey', f'frame-{frame_count}.jpg'), gray)
    cv2.imwrite(os.path.join(output_dir, 'morphology', f'frame-{frame_count}.jpg'), closing)
    cv2.imwrite(os.path.join(output_dir, 'noise_reduction', f'frame-{frame_count}.jpg'), blurred)
    cv2.imwrite(os.path.join(output_dir, 'hist_equilization', f'frame-{frame_count}.jpg'), equalized)
    cv2.imwrite(os.path.join(output_dir, 'contrast_improv', f'frame-{frame_count}.jpg'), contrasted)
    
    return gray

# Create output directories if they don't exist
output_dir = 'output'
os.makedirs(os.path.join(output_dir, 'original'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'sobel'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'grey'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'morphology'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'noise_reduction'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'hist_equilization'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'contrast_improv'), exist_ok=True)

# Open the video file
cap = cv2.VideoCapture(0)

# Initialize variables
frame_count = 0
fps = cap.get(cv2.CAP_PROP_FPS)
interval = int(fps) * 1  # Extract frame every 1 second

# Function to process frames and save processed frames every 1 second
def process_frames_and_save():
    global frame_count  # Declare frame_count as global
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        if frame_count % interval == 0:
            processed_frame = process_frame(frame, frame_count, output_dir)

# Start a thread for processing frames and saving processed frames
processing_thread = Thread(target=process_frames_and_save)
processing_thread.start()

# Start detecting tiredness
while cap.isOpened():
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
        
        # Draw eyes contours
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 255), 1)  # Yellow color
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 255), 1)  # Yellow color

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

        # Check if eyes are closed
        if ear < 0.25:
            alert_count += 1
            if alert_count >= 10 and not pygame.mixer.get_busy():
                # Visual alert (draw text)
                cv2.putText(frame, "ALERT! Fatigue Detected !!!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)

                # Play sound alert
                alert_sound.play()

        else:
            alert_count = 0

    # Display the frame (if needed)
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

    # Increment frame count
    frame_count += 1
    
    # Check if it's time to save EAR values to CSV
    if frame_count == 10:  # Save EAR values every 10 frames
        save_ear_values(ear_values)
        frame_count = 0
        ear_values = []
        
# Wait for the processing thread to finish
processing_thread.join()

# Release resources
cap.release()
cv2.destroyAllWindows()
