import cv2
import dlib
import numpy as np
from collections import deque
import time

# Load dlib's face detector and 68 landmarks model
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("./model/shape_predictor_68_face_landmarks.dat")

# Constants for lip movement detection
LIP_MOVEMENT_THRESHOLD = 3.0  # Threshold for detecting significant lip movement
HISTORY_SIZE = 10  # Number of frames to keep in history for smoothing
TALKING_FRAMES_THRESHOLD = 5  # Number of frames with movement to consider as talking

# Initialize history for lip distances
lip_distance_history = deque(maxlen=HISTORY_SIZE)
talking_frames_count = 0
last_talking_time = 0

def get_lip_height(landmarks):
    """Calculate the height of the lips (distance between upper and lower lip)"""
    # Upper lip points (51 is top of upper lip, 57 is bottom of lower lip)
    upper_lip = (landmarks.part(51).x, landmarks.part(51).y)
    lower_lip = (landmarks.part(57).x, landmarks.part(57).y)
    
    # Calculate Euclidean distance
    return np.sqrt((upper_lip[0] - lower_lip[0])**2 + (upper_lip[1] - lower_lip[1])**2)

def get_lip_aspect_ratio(landmarks):
    """Calculate the lip aspect ratio (width/height)"""
    # Lip width (points 48 and 54 are corners of mouth)
    lip_width = np.sqrt((landmarks.part(48).x - landmarks.part(54).x)**2 + 
                        (landmarks.part(48).y - landmarks.part(54).y)**2)
    
    # Lip height (already calculated in get_lip_height)
    lip_height = get_lip_height(landmarks)
    
    if lip_height > 0:
        return lip_width / lip_height
    return 0

def detect_lip_movement(current_height, avg_height):
    """Detect if there is significant lip movement"""
    return abs(current_height - avg_height) > LIP_MOVEMENT_THRESHOLD

def process_lip_movement(frame):
    """Process the frame to detect lip movement and talking"""
    global talking_frames_count, last_talking_time
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    
    is_talking = False
    lip_movement_detected = False
    
    for face in faces:
        landmarks = predictor(gray, face)
        
        # Get lip landmarks
        lip_points = []
        for i in range(48, 68):  # Landmarks 48-67 correspond to the mouth region
            x, y = landmarks.part(i).x, landmarks.part(i).y
            lip_points.append((x, y))
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)  # Draw the landmark
        
        # Connect the lip landmarks to visualize the mouth
        for i in range(len(lip_points) - 1):
            cv2.line(frame, lip_points[i], lip_points[i + 1], (0, 255, 0), 1)
        # Connect the last point to the first point
        cv2.line(frame, lip_points[-1], lip_points[0], (0, 255, 0), 1)
        
        # Calculate current lip height
        current_lip_height = get_lip_height(landmarks)
        lip_distance_history.append(current_lip_height)
        
        # Calculate average lip height from history
        if len(lip_distance_history) > 0:
            avg_lip_height = np.mean(lip_distance_history)
            
            # Detect lip movement
            lip_movement_detected = detect_lip_movement(current_lip_height, avg_lip_height)
            
            # Update talking state
            if lip_movement_detected:
                talking_frames_count += 1
                if talking_frames_count >= TALKING_FRAMES_THRESHOLD:
                    is_talking = True
                    last_talking_time = time.time()
            else:
                # Reset talking frames count if no movement for a while
                if time.time() - last_talking_time > 1.0:  # 1 second without movement
                    talking_frames_count = max(0, talking_frames_count - 1)
                    if talking_frames_count < TALKING_FRAMES_THRESHOLD:
                        is_talking = False
        
        # Draw lip aspect ratio and talking status
        lip_aspect_ratio = get_lip_aspect_ratio(landmarks)
        cv2.putText(frame, f"Lip AR: {lip_aspect_ratio:.2f}", 
                   (face.left(), face.bottom() + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    return frame, is_talking
