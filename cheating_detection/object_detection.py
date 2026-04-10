import cv2
import torch
from ultralytics import YOLO
import time

# Load pre-trained YOLO model for object detection
model = YOLO("yolov8n.pt")  # Using the nano model for speed
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Define suspicious objects for exams
SUSPICIOUS_OBJECTS = [
    'book', 'cell phone', 'laptop', 'keyboard', 'mouse', 'remote', 'tv', 'monitor',
    'tablet', 'paper', 'notebook', 'pen', 'pencil'
]

# Constants for object detection
CONFIDENCE_THRESHOLD = 0.5
DETECTION_PERSISTENCE = 3.0  # How long to show a detection after it disappears (seconds)

# Initialize variables
last_detection_time = {}  # Track when each object was last detected

def process_object_detection(frame):
    """Process the frame to detect suspicious objects"""
    global last_detection_time
    
    # Run inference
    results = model(frame, verbose=False)
    
    # Track detected objects
    detected_objects = {}
    suspicious_objects_detected = False
    
    current_time = time.time()
    
    for result in results:
        for box in result.boxes:
            conf = box.conf[0].item()
            cls = int(box.cls[0].item())
            cls_name = model.names[cls]
            
            # Skip if confidence is too low
            if conf < CONFIDENCE_THRESHOLD:
                continue
            
            # Check if this is a suspicious object
            is_suspicious = cls_name.lower() in SUSPICIOUS_OBJECTS
            
            # Update detection time
            if is_suspicious:
                last_detection_time[cls_name] = current_time
                suspicious_objects_detected = True
            
            # Add to detected objects
            if cls_name in detected_objects:
                detected_objects[cls_name] += 1
            else:
                detected_objects[cls_name] = 1
            
            # Draw bounding box
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            color = (0, 0, 255) if is_suspicious else (0, 255, 0)
            label = f"{cls_name} ({conf:.2f})"
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Check for recently detected objects that are no longer visible
    for obj, last_time in list(last_detection_time.items()):
        if current_time - last_time < DETECTION_PERSISTENCE:
            if obj not in detected_objects:
                # Object was recently detected but is no longer visible
                cv2.putText(frame, f"Recently detected: {obj}", (20, 210), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                suspicious_objects_detected = True
        else:
            # Remove old detections
            last_detection_time.pop(obj)
    
    # Display warning if suspicious objects detected
    if suspicious_objects_detected:
        cv2.putText(frame, "WARNING: Suspicious objects detected!", (20, 240), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    return frame, suspicious_objects_detected, detected_objects
