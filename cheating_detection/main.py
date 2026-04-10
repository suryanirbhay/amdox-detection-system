import cv2
import time
import os
import argparse
from eye_movement import process_eye_movement
from head_pose import process_head_pose
from mobile_detection import process_mobile_detection
from lip_movement import process_lip_movement
from facial_expression import process_facial_expression
from person_detection import process_person_detection
from object_detection import process_object_detection
from behavior_analysis import process_behavior_analysis, load_training_data

# Initialize webcam
cap = cv2.VideoCapture(0)

# Create a log directory for screenshots
log_dir = "log"
os.makedirs(log_dir, exist_ok=True)

# Calibration for head pose
calibrated_angles = None
start_time = time.time()

# Timers for each functionality
head_misalignment_start_time = None
eye_misalignment_start_time = None
mobile_detection_start_time = None
talking_detection_start_time = None
expression_detection_start_time = None
person_detection_start_time = None
object_detection_start_time = None

# Previous states
previous_head_state = "Looking at Screen"
previous_eye_state = "Looking at Screen"
previous_mobile_state = False
previous_talking_state = False
previous_expression_state = "Neutral"
previous_person_count = 1
previous_object_state = False

# Initialize default values
head_direction = "Looking at Screen"
is_talking = False
facial_expression = "Neutral"
person_count = 1
multiple_people = False
new_person_entered = False
suspicious_objects = False

# Initialize behavior analysis
behavior_analysis_start_time = None
previous_behavior = "Normal"

# Try to load pre-trained behavior model
load_training_data()

# Parse command line arguments
parser = argparse.ArgumentParser(description='Cheating Detection System with Behavior Analysis')
parser.add_argument('--train', action='store_true', help='Run in training mode to collect behavior data')
args = parser.parse_args()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Process eye movement
    frame, gaze_direction = process_eye_movement(frame)
    cv2.putText(frame, f"Gaze Direction: {gaze_direction}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Process head pose
    if time.time() - start_time <= 5:  # Calibration time
        cv2.putText(frame, "Calibrating... Keep your head straight", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        if calibrated_angles is None:
            _, calibrated_angles = process_head_pose(frame, None)
    else:
        frame, head_direction = process_head_pose(frame, calibrated_angles)
        cv2.putText(frame, f"Head Direction: {head_direction}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Process mobile detection
    frame, mobile_detected = process_mobile_detection(frame)
    cv2.putText(frame, f"Mobile Detected: {mobile_detected}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Process lip movement detection
    frame, is_talking = process_lip_movement(frame)
    cv2.putText(frame, f"Talking: {is_talking}", (20, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Process facial expression detection
    frame, facial_expression = process_facial_expression(frame)
    cv2.putText(frame, f"Expression: {facial_expression}", (20, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Process person detection
    frame, person_count, multiple_people, new_person_entered = process_person_detection(frame)

    # Process object detection
    frame, suspicious_objects, detected_objects = process_object_detection(frame)

    # Prepare data for behavior analysis
    head_data = {
        'direction': head_direction,
        'rapid_movement': head_direction == "Rapid Movement",
        'pitch': getattr(process_head_pose, 'last_pitch', 0),
        'yaw': getattr(process_head_pose, 'last_yaw', 0),
        'roll': getattr(process_head_pose, 'last_roll', 0)
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
        'detected_objects': detected_objects
    }

    # Process behavior analysis
    frame, behavior_result = process_behavior_analysis(
        head_data, eye_data, lip_data, expression_data, person_data, object_data, frame
    )

    # Extract behavior information
    behavior = behavior_result['behavior']
    confidence = behavior_result['confidence']

    # Check for head misalignment
    if head_direction != "Looking at Screen":
        if head_misalignment_start_time is None:
            head_misalignment_start_time = time.time()
        elif time.time() - head_misalignment_start_time >= 3:
            filename = os.path.join(log_dir, f"head_{head_direction}_{int(time.time())}.png")
            cv2.imwrite(filename, frame)
            print(f"Screenshot saved: {filename}")
            head_misalignment_start_time = None  # Reset timer
    else:
        head_misalignment_start_time = None  # Reset timer

    # Check for eye misalignment
    if gaze_direction != "Looking at Screen":
        if eye_misalignment_start_time is None:
            eye_misalignment_start_time = time.time()
        elif time.time() - eye_misalignment_start_time >= 3:
            filename = os.path.join(log_dir, f"eye_{gaze_direction}_{int(time.time())}.png")
            cv2.imwrite(filename, frame)
            print(f"Screenshot saved: {filename}")
            eye_misalignment_start_time = None  # Reset timer
    else:
        eye_misalignment_start_time = None  # Reset timer

    # Check for mobile detection
    if mobile_detected:
        if mobile_detection_start_time is None:
            mobile_detection_start_time = time.time()
        elif time.time() - mobile_detection_start_time >= 3:
            filename = os.path.join(log_dir, f"mobile_detected_{int(time.time())}.png")
            cv2.imwrite(filename, frame)
            print(f"Screenshot saved: {filename}")
            mobile_detection_start_time = None  # Reset timer
    else:
        mobile_detection_start_time = None  # Reset timer

    # Check for talking detection
    if is_talking:
        if talking_detection_start_time is None:
            talking_detection_start_time = time.time()
        elif time.time() - talking_detection_start_time >= 3:
            filename = os.path.join(log_dir, f"talking_{int(time.time())}.png")
            cv2.imwrite(filename, frame)
            print(f"Screenshot saved: {filename}")
            talking_detection_start_time = None  # Reset timer
    else:
        talking_detection_start_time = None  # Reset timer

    # Check for suspicious facial expressions
    if facial_expression != "Neutral":
        if expression_detection_start_time is None:
            expression_detection_start_time = time.time()
        elif time.time() - expression_detection_start_time >= 3:
            filename = os.path.join(log_dir, f"expression_{facial_expression}_{int(time.time())}.png")
            cv2.imwrite(filename, frame)
            print(f"Screenshot saved: {filename}")
            expression_detection_start_time = None  # Reset timer
    else:
        expression_detection_start_time = None  # Reset timer

    # Check for multiple people or new person
    if multiple_people or new_person_entered:
        if person_detection_start_time is None:
            person_detection_start_time = time.time()
        elif time.time() - person_detection_start_time >= 2:
            filename = os.path.join(log_dir, f"person_count_{person_count}_{int(time.time())}.png")
            cv2.imwrite(filename, frame)
            print(f"Screenshot saved: {filename}")
            person_detection_start_time = None  # Reset timer
    else:
        person_detection_start_time = None  # Reset timer

    # Check for suspicious objects
    if suspicious_objects:
        if object_detection_start_time is None:
            object_detection_start_time = time.time()
        elif time.time() - object_detection_start_time >= 3:
            filename = os.path.join(log_dir, f"suspicious_objects_{int(time.time())}.png")
            cv2.imwrite(filename, frame)
            print(f"Screenshot saved: {filename}")
            object_detection_start_time = None  # Reset timer
    else:
        object_detection_start_time = None  # Reset timer

    # Check for suspicious behavior
    if behavior == "Suspicious" and confidence > 0.6:
        if behavior_analysis_start_time is None:
            behavior_analysis_start_time = time.time()
        elif time.time() - behavior_analysis_start_time >= 2:
            filename = os.path.join(log_dir, f"suspicious_behavior_{int(time.time())}.png")
            cv2.imwrite(filename, frame)
            print(f"Screenshot saved: {filename}")
            behavior_analysis_start_time = None  # Reset timer
    else:
        behavior_analysis_start_time = None  # Reset timer

    # Display the combined output
    cv2.imshow("Combined Detection", frame)

    # Handle keyboard input
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('n') and args.train:
        # Mark current behavior as normal for training
        from behavior_analysis import add_training_sample
        add_training_sample(behavior_result['features'], False)
        print("Added normal behavior sample")
    elif key == ord('s') and args.train:
        # Mark current behavior as suspicious for training
        from behavior_analysis import add_training_sample
        add_training_sample(behavior_result['features'], True)
        print("Added suspicious behavior sample")

# Save any collected training data
if args.train:
    from behavior_analysis import save_training_data
    save_training_data()
    print("Training data saved")

cap.release()
cv2.destroyAllWindows()

# If this was run in training mode, print instructions
if args.train:
    print("\nTraining mode completed.")
    print("You can now run the system normally to use the trained behavior model.")