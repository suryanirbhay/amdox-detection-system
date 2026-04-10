import pickle
import os

# Create model directory if it doesn't exist
os.makedirs("model", exist_ok=True)

# Create an empty list for training data
training_data = []

# Save the empty list to a pickle file
with open("model/behavior_training_data.pkl", "wb") as f:
    pickle.dump(training_data, f)

print("Created empty training data file")
