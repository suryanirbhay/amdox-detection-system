import numpy as np
import time
import pickle
import os
import cv2
from collections import deque
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

# Constants for behavior analysis
HISTORY_SIZE = 100  # Number of frames to keep in history
FEATURES_PER_FRAME = 15  # Number of features extracted per frame
MODEL_PATH = "./model/behavior_model.pkl"
SCALER_PATH = "./model/behavior_scaler.pkl"

# Initialize data structures
behavior_history = deque(maxlen=HISTORY_SIZE)
feature_history = deque(maxlen=HISTORY_SIZE)
last_prediction_time = 0
prediction_cooldown = 1.0  # Seconds between predictions

# Initialize model and scaler
model = None
scaler = None

# Try to load pre-trained model and scaler if they exist
try:
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        print("Loaded pre-trained behavior model and scaler")
except Exception as e:
    print(f"Could not load behavior model: {e}")

def extract_features(head_data, eye_data, lip_data, expression_data, person_data, object_data):
    """Extract features from various detection modules"""
    features = []

    # Head pose features
    head_direction = head_data.get('direction', 'Looking at Screen')
    head_rapid_movement = head_data.get('rapid_movement', False)
    pitch = head_data.get('pitch', 0)
    yaw = head_data.get('yaw', 0)
    roll = head_data.get('roll', 0)

    # Convert head direction to numerical value
    head_dir_map = {
        'Looking at Screen': 0,
        'Looking Left': 1,
        'Looking Right': 2,
        'Looking Up': 3,
        'Looking Down': 4,
        'Tilted': 5,
        'Rapid Movement': 6
    }
    head_dir_value = head_dir_map.get(head_direction, 0)

    # Eye movement features
    gaze_direction = eye_data.get('direction', 'Looking Center')

    # Convert gaze direction to numerical value
    gaze_dir_map = {
        'Looking Center': 0,
        'Looking Left': 1,
        'Looking Right': 2,
        'Looking Up': 3,
        'Looking Down': 4
    }
    gaze_dir_value = gaze_dir_map.get(gaze_direction, 0)

    # Lip movement features
    is_talking = 1 if lip_data.get('is_talking', False) else 0

    # Facial expression features
    expression = expression_data.get('expression', 'Neutral')

    # Convert expression to numerical value
    expr_map = {
        'Neutral': 0,
        'Confused': 1,
        'Smiling': 2,
        'Suspicious (Confused+Smiling)': 3
    }
    expr_value = expr_map.get(expression, 0)

    # Person detection features
    person_count = person_data.get('count', 1)
    multiple_people = 1 if person_data.get('multiple_people', False) else 0
    new_person = 1 if person_data.get('new_person', False) else 0

    # Object detection features
    suspicious_objects = 1 if object_data.get('suspicious_objects', False) else 0
    object_count = len(object_data.get('detected_objects', {}))

    # Combine all features
    features = [
        head_dir_value,
        1 if head_rapid_movement else 0,
        pitch,
        yaw,
        roll,
        gaze_dir_value,
        is_talking,
        expr_value,
        person_count,
        multiple_people,
        new_person,
        suspicious_objects,
        object_count,
        # Add time-based features
        time.time() % 86400 / 86400,  # Time of day normalized to [0,1]
        len(feature_history) / HISTORY_SIZE  # Progress through current session
    ]

    return features

def train_model(features, labels):
    """Train a new behavior model with collected data"""
    global model, scaler

    if len(features) < 10:
        print("Not enough data to train model")
        return False

    # Convert to numpy arrays
    X = np.array(features)
    y = np.array(labels)

    # Create and fit scaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Create and train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)

    # Save model and scaler
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    print(f"Trained behavior model with {len(features)} samples")
    return True

def predict_behavior(features):
    """Predict behavior using the trained model"""
    global model, scaler

    if model is None or scaler is None:
        return "Unknown", 0.0

    # Scale features
    features_scaled = scaler.transform([features])

    # Get prediction and probability
    prediction = model.predict(features_scaled)[0]
    probabilities = model.predict_proba(features_scaled)[0]
    confidence = max(probabilities)

    behavior_map = {
        0: "Normal",
        1: "Suspicious"
    }

    return behavior_map.get(prediction, "Unknown"), confidence

def add_training_sample(features, is_suspicious):
    """Add a labeled sample for training"""
    feature_history.append((features, 1 if is_suspicious else 0))

    # If we have enough samples, train the model
    if len(feature_history) >= HISTORY_SIZE:
        X = [f for f, _ in feature_history]
        y = [l for _, l in feature_history]
        train_model(X, y)

def process_behavior_analysis(head_data, eye_data, lip_data, expression_data, person_data, object_data, frame):
    """Process all detection data to analyze behavior"""
    global last_prediction_time

    # Extract features from all detection modules
    features = extract_features(head_data, eye_data, lip_data, expression_data, person_data, object_data)

    # Add features to history
    behavior_history.append(features)

    # Only make predictions at certain intervals to avoid flickering
    current_time = time.time()
    if current_time - last_prediction_time < prediction_cooldown:
        # Return last prediction if we're in cooldown
        if hasattr(process_behavior_analysis, 'last_result'):
            return frame, process_behavior_analysis.last_result
        else:
            return frame, {"behavior": "Normal", "confidence": 0.0, "features": features}

    last_prediction_time = current_time

    # If we have a trained model, make a prediction
    if model is not None and scaler is not None:
        behavior, confidence = predict_behavior(features)
    else:
        # Simple rule-based detection if no model is available
        suspicious_count = 0
        if head_data.get('direction', 'Looking at Screen') != 'Looking at Screen':
            suspicious_count += 1
        if head_data.get('rapid_movement', False):
            suspicious_count += 2
        if eye_data.get('direction', 'Looking Center') != 'Looking Center':
            suspicious_count += 1
        if lip_data.get('is_talking', False):
            suspicious_count += 1
        if expression_data.get('expression', 'Neutral') != 'Neutral':
            suspicious_count += 1
        if person_data.get('multiple_people', False):
            suspicious_count += 3
        if person_data.get('new_person', False):
            suspicious_count += 2
        if object_data.get('suspicious_objects', False):
            suspicious_count += 2

        behavior = "Suspicious" if suspicious_count >= 3 else "Normal"
        confidence = min(1.0, suspicious_count / 10.0)

    # Draw behavior analysis on frame
    cv2.putText(frame, f"Behavior: {behavior} ({confidence:.2f})",
               (20, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
               (0, 0, 255) if behavior == "Suspicious" else (0, 255, 0), 2)

    # Store result for cooldown period
    result = {"behavior": behavior, "confidence": confidence, "features": features}
    process_behavior_analysis.last_result = result

    return frame, result

def save_training_data(file_path="./model/behavior_training_data.pkl"):
    """Save collected training data for future use"""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'wb') as f:
        pickle.dump(list(feature_history), f)
    print(f"Saved {len(feature_history)} training samples to {file_path}")

def load_training_data(file_path="./model/behavior_training_data.pkl"):
    """Load training data from file"""
    global feature_history

    if not os.path.exists(file_path):
        print(f"Training data file {file_path} not found")
        return False

    with open(file_path, 'rb') as f:
        feature_history = deque(pickle.load(f), maxlen=HISTORY_SIZE)

    print(f"Loaded {len(feature_history)} training samples from {file_path}")

    # Train model with loaded data
    if len(feature_history) > 0:
        X = [f for f, _ in feature_history]
        y = [l for _, l in feature_history]
        train_model(X, y)

    return True
