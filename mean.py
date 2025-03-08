import json
import numpy as np
import torch
from pathlib import Path
from datasets.video.utils import read_video  # your read_video function

# Path to the training JSON file
train_json_path = "/home/Gurjot/dfot_data/labels/train.json"
videos_dir = Path("/home/Gurjot/dfot_data/20bn-something-something-v2")

# Load training annotations
with open(train_json_path, "r") as f:
    train_annotations = json.load(f)

# Initialize accumulators for 3 channels (assuming RGB)
sum_channels = np.zeros(3)
sum_sq_channels = np.zeros(3)
num_pixels = 0

for ann in train_annotations:
    video_id = ann["id"]
    print(video_id)
    video_file = videos_dir / f"{video_id}.webm"
    if not video_file.exists():
        print(f"Warning: {video_file} not found!")
        continue
    
    # Load video frames in THWC format and convert to float
    frames = read_video(str(video_file), output_format="THWC").float()
    # Scale pixel values to [0, 1] (if that's your training scale)
    frames = frames / 255.0
    
    # Get shape (T, H, W, C) and reshape to (-1, C)
    T, H, W, C = frames.shape
    frames_np = frames.view(-1, C).numpy()  # Convert to numpy for easier summing
    
    sum_channels += frames_np.sum(axis=0)
    sum_sq_channels += (frames_np ** 2).sum(axis=0)
    num_pixels += frames_np.shape[0]

# Compute mean and std for each channel
mean = sum_channels / num_pixels
variance = sum_sq_channels / num_pixels - mean ** 2
std = np.sqrt(variance)

print("Training Mean per channel:", mean)
print("Training Std per channel:", std)