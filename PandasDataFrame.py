import os
import pandas as pd

# Automatically find the latest 'exp' folder (e.g., exp3, exp4, etc.)
base_dir = 'runs/detect'  # Base directory where exp folders are stored
exp_folders = sorted([folder for folder in os.listdir(base_dir) if folder.startswith('exp')], reverse=True)

# If exp folders exist, get the latest one
if exp_folders:
    latest_exp = exp_folders[0]
    results_dir = os.path.join(base_dir, latest_exp, 'labels')
else:
    raise FileNotFoundError("No exp folders found in the runs/detect directory.")

# Define class mappings (for example, assuming '2' maps to 'car')
class_mapping = {
    0: 'person',
    1: 'bicycle',
    2: 'car',
    3: 'motorbike',
    4: 'bus',
    # Add more class mappings as per your YOLO model
}

# List to hold the parsed data
detection_data = []

# Iterate through all text files in the labels folder
for filename in os.listdir(results_dir):
    if filename.endswith('.txt'):
        frame_id = filename.split('.')[0]  # Get the frame ID (e.g., 'frame_0001')
        with open(os.path.join(results_dir, filename), 'r') as file:
            for line in file:
                try:
                    parts = line.strip().split()
                    class_id = int(parts[0])
                    confidence = float(parts[1])
                    # Parse bounding box values
                    x_center, y_center, width, height = map(float, parts[2:])
                    
                    # Convert class ID to class name
                    class_name = class_mapping.get(class_id, "Unknown")
                    
                    # Store the detection info
                    detection_data.append([frame_id, class_name, confidence, x_center, y_center, width, height])
                except ValueError:
                    # Skip lines that don't match the expected format
                    print(f"Error parsing bounding box values in {filename}: {line.strip()}")
                    continue

# Convert the data to a DataFrame
df = pd.DataFrame(detection_data, columns=['frame', 'class', 'confidence', 'x_center', 'y_center', 'width', 'height'])

# Display the first few rows of the DataFrame
print(df.head())

# Optionally, save the DataFrame to a CSV file for further analysis
df.to_csv('detection_results.csv', index=False)
