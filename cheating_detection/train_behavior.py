"""
Behavior Training Script for Cheating Detection System

This script runs the main application in training mode, allowing you to collect
labeled examples of normal and suspicious behaviors to train the machine learning model.

Usage:
    python train_behavior.py

Instructions:
    - Press 'n' to mark current behavior as normal
    - Press 's' to mark current behavior as suspicious
    - Press 'q' to quit and save the collected data
"""

import os
import sys

if __name__ == "__main__":
    # Run the main application with the --train flag
    os.system("python main.py --train")
