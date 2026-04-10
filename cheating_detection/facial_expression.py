import cv2
import dlib
import numpy as np
from collections import deque
import time

# Load dlib's face detector and 68 landmarks model
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("./model/shape_predictor_68_face_landmarks.dat")

# Constants for facial expression detection
BROW_MOVEMENT_THRESHOLD = 5.0  # Threshold for detecting significant eyebrow movement
SMILE_THRESHOLD = 2.5  # Threshold for detecting smiles
HISTORY_SIZE = 10  # Number of frames to keep in history for smoothing

# Initialize history for facial measurements
brow_distance_history = deque(maxlen=HISTORY_SIZE)
mouth_aspect_ratio_history = deque(maxlen=HISTORY_SIZE)

def get_brow_distance(landmarks):
    """Calculate the distance between eyebrows and eyes"""
    # Left eyebrow (points 17-21) to left eye (points 36-41)
    left_brow = np.mean([(landmarks.part(i).y) for i in range(17, 22)])
    left_eye = np.mean([(landmarks.part(i).y) for i in range(36, 42)])
    
    # Right eyebrow (points 22-26) to right eye (points 42-47)
    right_brow = np.mean([(landmarks.part(i).y) for i in range(22, 27)])
    right_eye = np.mean([(landmarks.part(i).y) for i in range(42, 48)])
    
    # Average distance
    return np.mean([left_eye - left_brow, right_eye - right_brow])

def get_mouth_aspect_ratio(landmarks):
    """Calculate the mouth aspect ratio (width/height)"""
    # Horizontal distance (points 48 and 54 are corners of mouth)
    mouth_width = np.sqrt((landmarks.part(54).x - landmarks.part(48).x)**2 + 
                         (landmarks.part(54).y - landmarks.part(48).y)**2)
    
    # Vertical distance (points 51 and 57 are top and bottom of inner lips)
    mouth_height = np.sqrt((landmarks.part(57).x - landmarks.part(51).x)**2 + 
                          (landmarks.part(57).y - landmarks.part(51).y)**2)
    
    # Return aspect ratio
    if mouth_height > 0:
        return mouth_width / mouth_height
    return 0

def detect_confusion(brow_distance, avg_brow_distance):
    """Detect if the person looks confused (raised eyebrows)"""
    return brow_distance > avg_brow_distance + BROW_MOVEMENT_THRESHOLD

def detect_smile(mouth_ar, avg_mouth_ar):
    """Detect if the person is smiling"""
    return mouth_ar > avg_mouth_ar + SMILE_THRESHOLD

def process_facial_expression(frame):
    """Process the frame to detect facial expressions"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    
    is_confused = False
    is_smiling = False
    
    for face in faces:
        landmarks = predictor(gray, face)
        
        # Calculate facial measurements
        brow_distance = get_brow_distance(landmarks)
        mouth_ar = get_mouth_aspect_ratio(landmarks)
        
        # Update history
        brow_distance_history.append(brow_distance)
        mouth_aspect_ratio_history.append(mouth_ar)
        
        # Calculate averages from history
        if len(brow_distance_history) > 0 and len(mouth_aspect_ratio_history) > 0:
            avg_brow_distance = np.mean(brow_distance_history)
            avg_mouth_ar = np.mean(mouth_aspect_ratio_history)
            
            # Detect expressions
            is_confused = detect_confusion(brow_distance, avg_brow_distance)
            is_smiling = detect_smile(mouth_ar, avg_mouth_ar)
            
            # Draw facial landmarks for eyebrows and mouth
            for i in range(17, 27):  # Eyebrows
                x, y = landmarks.part(i).x, landmarks.part(i).y
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
            
            for i in range(48, 68):  # Mouth
                x, y = landmarks.part(i).x, landmarks.part(i).y
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
            
            # Draw measurements on frame
            cv2.putText(frame, f"Brow Dist: {brow_distance:.2f}", 
                       (face.left(), face.bottom() + 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            cv2.putText(frame, f"Mouth AR: {mouth_ar:.2f}", 
                       (face.left(), face.bottom() + 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    
    # Determine overall expression
    expression = "Neutral"
    if is_confused and is_smiling:
        expression = "Suspicious (Confused+Smiling)"
    elif is_confused:
        expression = "Confused"
    elif is_smiling:
        expression = "Smiling"
    
    return frame, expression
