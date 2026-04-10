import cv2
import time
import os
import numpy as np
import keyboard
from behavior_analysis import extract_features, add_training_sample, save_training_data, load_training_data, train_model
from eye_movement import process_eye_movement
from head_pose import process_head_pose
from lip_movement import process_lip_movement
from facial_expression import process_facial_expression
from person_detection import process_person_detection
from object_detection import process_object_detection

def collect_training_data():
    """
    Collect training data for behavior analysis.
    Press 'n' to mark normal behavior.
    Press 's' to mark suspicious behavior.
    Press 'q' to quit and save the collected data.
    """
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    # Create a log directory for screenshots
    log_dir = "training_data"
    os.makedirs(log_dir, exist_ok=True)
    
    # Calibration for head pose
    calibrated_angles = None
    start_time = time.time()
    
    # Initialize default values
    head_direction = "Looking at Screen"
    is_talking = False
    facial_expression = "Neutral"
    person_count = 1
    multiple_people = False
    new_person_entered = False
    suspicious_objects = False
    
    # Load existing training data if available
    load_training_data()
    
    print("Starting training data collection...")
    print("Press 'n' to mark normal behavior")
    print("Press 's' to mark suspicious behavior")
    print("Press 'q' to quit and save the collected data")
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Process eye movement
        frame, gaze_direction = process_eye_movement(frame)
        
        # Process head pose
        if time.time() - start_time <= 5:  # Calibration time
            cv2.putText(frame, "Calibrating... Keep your head straight", (50, 200), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            if calibrated_angles is None:
                _, calibrated_angles = process_head_pose(frame, None)
        else:
            frame, head_direction = process_head_pose(frame, calibrated_angles)
        
        # Process lip movement detection
        frame, is_talking = process_lip_movement(frame)
        
        # Process facial expression detection
        frame, facial_expression = process_facial_expression(frame)
        
        # Process person detection
        frame, person_count, multiple_people, new_person_entered = process_person_detection(frame)
        
        # Process object detection (only every 10 frames to save processing time)
        if frame_count % 10 == 0:
            frame, suspicious_objects, detected_objects = process_object_detection(frame)
        
        # Prepare data for feature extraction
        head_data = {
            'direction': head_direction,
            'rapid_movement': head_direction == "Rapid Movement",
            'pitch': 0,  # These would come from head_pose.py
            'yaw': 0,
            'roll': 0
        }
        
        eye_data = {
            'direction': gaze_direction
        }
        
        lip_data = {
            'is_talking': is_talking
        }
        
        expression_data = {
            'expression': facial_expression
        }
        
        person_data = {
            'count': person_count,
            'multiple_people': multiple_people,
            'new_person': new_person_entered
        }
        
        object_data = {
            'suspicious_objects': suspicious_objects,
            'detected_objects': getattr(process_object_detection, 'last_objects', {})
        }
        
        # Extract features
        features = extract_features(head_data, eye_data, lip_data, expression_data, person_data, object_data)
        
        # Check for key presses to label data
        if keyboard.is_pressed('n'):
            add_training_sample(features, False)
            print("Added normal behavior sample")
            # Take a screenshot for reference
            filename = os.path.join(log_dir, f"normal_{int(time.time())}.png")
            cv2.imwrite(filename, frame)
            time.sleep(0.5)  # Prevent multiple samples from one key press
        
        if keyboard.is_pressed('s'):
            add_training_sample(features, True)
            print("Added suspicious behavior sample")
            # Take a screenshot for reference
            filename = os.path.join(log_dir, f"suspicious_{int(time.time())}.png")
            cv2.imwrite(filename, frame)
            time.sleep(0.5)  # Prevent multiple samples from one key press
        
        # Display instructions on frame
        cv2.putText(frame, "Press 'n' for normal behavior", (20, 360), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, "Press 's' for suspicious behavior", (20, 390), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, "Press 'q' to quit and save", (20, 420), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Display the frame
        cv2.imshow("Behavior Training", frame)
        
        # Check for quit
        if cv2.waitKey(1) & 0xFF == ord('q') or keyboard.is_pressed('q'):
            break
    
    # Save collected training data
    save_training_data()
    
    cap.release()
    cv2.destroyAllWindows()
    print("Training data collection completed and saved.")

if __name__ == "__main__":
    collect_training_data()
