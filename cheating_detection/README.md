# Enhanced Cheating Detection System

This system uses computer vision and machine learning to detect potential cheating behaviors during exams or assessments.

## Features

- **Eye Movement Detection**: Tracks where the user is looking
- **Head Pose Detection**: Monitors head position and detects rapid movements
- **Lip Movement Detection**: Detects if the user is talking
- **Facial Expression Analysis**: Identifies suspicious expressions
- **Person Detection**: Counts people in the frame and detects if someone new enters
- **Object Detection**: Identifies suspicious objects like phones, books, etc.
- **Behavior Analysis**: Uses machine learning to identify suspicious behavior patterns

## Requirements

- Python 3.7+
- OpenCV
- dlib
- NumPy
- scikit-learn
- PyTorch
- Ultralytics (YOLOv8)
- keyboard

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/Cheating-Surveillance-System.git
   cd Cheating-Surveillance-System
   ```

2. Install the required dependencies:
   ```
   pip install opencv-python dlib numpy scikit-learn torch ultralytics keyboard
   ```

3. Download the required models:
   - Download the shape_predictor_68_face_landmarks.dat file from the dlib website and place it in the `model` directory
   - The YOLOv8 model will be downloaded automatically on first run

## Usage

### Running the Detection System

To run the standard detection system:

```
python main.py
```

The system will:
1. Calibrate to your head position (keep your head straight and looking at the screen for the first 5 seconds)
2. Start monitoring for suspicious behaviors
3. Save screenshots to the `log` directory when suspicious activities are detected

### Training the Behavior Analysis Model

To train the behavior analysis model with your own examples:

```
python train_behavior.py
```

During training mode:
- Press 'n' to mark the current behavior as normal
- Press 's' to mark the current behavior as suspicious
- Press 'q' to quit and save the collected data

The more examples you provide, the better the model will become at distinguishing between normal and suspicious behaviors.

## Components

- `main.py`: Main application that integrates all detection modules
- `eye_movement.py`: Detects eye movements and gaze direction
- `head_pose.py`: Tracks head position and detects rapid movements
- `lip_movement.py`: Detects lip movements and talking
- `facial_expression.py`: Analyzes facial expressions
- `person_detection.py`: Counts people and detects new entries
- `object_detection.py`: Detects suspicious objects
- `behavior_analysis.py`: Machine learning for behavior pattern analysis
- `behavior_training.py`: Script to run the system in training mode

## Customization

You can adjust the sensitivity of various detection components by modifying the threshold values in their respective files.

## License

[Your License Information]
