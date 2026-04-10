import cv2
import numpy as np
import time

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Constants for person detection
CONFIDENCE_THRESHOLD = 0.4  # Lowered to improve detection sensitivity
HISTORY_SIZE = 15  # Increased to have more stable history
PERSON_ENTRY_THRESHOLD = 3  # Number of consecutive frames to confirm a new person
MIN_DETECTION_FRAMES = 5  # Minimum number of frames with detection to consider reliable

# Initialize variables
person_count_history = []
last_person_entry_time = 0
new_person_frames = 0
previous_person_count = 0

def process_person_detection(frame):
    """Process the frame to detect and count people"""
    global person_count_history, last_person_entry_time, new_person_frames, previous_person_count

    # Resize frame for faster detection
    frame_resized = cv2.resize(frame, (640, 480))

    # Detect people in the frame with improved parameters
    boxes, weights = hog.detectMultiScale(
        frame_resized,
        winStride=(4, 4),  # Smaller stride for better detection
        padding=(8, 8),    # Increased padding
        scale=1.03         # Smaller scale step for more detection opportunities
    )

    # Filter detections by confidence
    filtered_boxes = []
    for i, box in enumerate(boxes):
        if weights[i] > CONFIDENCE_THRESHOLD:
            filtered_boxes.append(box)

    # Scale boxes back to original frame size
    height_ratio = frame.shape[0] / 480
    width_ratio = frame.shape[1] / 640

    # Draw bounding boxes and count people
    person_count = 0
    for box in filtered_boxes:
        x, y, w, h = box
        x = int(x * width_ratio)
        y = int(y * height_ratio)
        w = int(w * width_ratio)
        h = int(h * height_ratio)

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f"Person {person_count + 1}", (x, y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        person_count += 1

    # Fallback detection using face detection if no people detected
    if person_count == 0:
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Use a pre-trained face detector
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)

        # If faces are detected, count them as people
        if len(faces) > 0:
            person_count = len(faces)

            # Draw face detections
            for i, (x, y, w, h) in enumerate(faces):
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(frame, f"Face {i + 1}", (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Update person count history
    person_count_history.append(person_count)
    if len(person_count_history) > HISTORY_SIZE:
        person_count_history.pop(0)

    # Calculate average person count to reduce false positives
    avg_person_count = int(np.ceil(np.mean(person_count_history)))

    # Ensure person count is at least 1 if we've previously detected someone
    # This prevents the count from dropping to 0 incorrectly
    if avg_person_count == 0 and previous_person_count >= 1:
        # If we've consistently seen a person before, assume they're still there
        # but temporarily not detected
        if sum(1 for count in person_count_history if count > 0) >= 2:
            avg_person_count = 1

    # Detect if a new person entered the frame
    new_person_entered = False
    if avg_person_count > previous_person_count:
        new_person_frames += 1
        if new_person_frames >= PERSON_ENTRY_THRESHOLD:
            new_person_entered = True
            last_person_entry_time = time.time()
    else:
        new_person_frames = 0

    # Update previous person count
    previous_person_count = avg_person_count

    # Display person count on frame
    cv2.putText(frame, f"Person Count: {avg_person_count}", (20, 120),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Display warning if multiple people detected
    multiple_people = avg_person_count > 1
    if multiple_people:
        cv2.putText(frame, "WARNING: Multiple people detected!", (20, 150),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Display warning if new person entered
    if new_person_entered or (time.time() - last_person_entry_time < 3.0 and avg_person_count > 1):
        cv2.putText(frame, "WARNING: New person entered the frame!", (20, 180),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    return frame, avg_person_count, multiple_people, new_person_entered

