# 🎾 Tennis Pose Estimation with YOLOv8

This project uses a custom-trained YOLOv8 pose model to estimate the body and racquet keypoints of a tennis player from video footage. The primary goal is to visualize and analyze movement mechanics frame-by-frame for technique analysis and improvement.

## 🔍 Overview

- 📹 Input: Standard tennis videos (e.g., `.mp4`)
- 🤖 Model: Custom-trained YOLOv8 pose model
- 🧠 Output: Visual overlay of keypoints per frame (13 keypoints per detected person)
- 🎯 Use Case: Tennis technique analysis, coaching, and motion feedback

## 📁 Project Structure

```bash
.
├── example1.mp4                # Sample tennis video input
├── runs/pose/tennis_pose_model11/
│   └── weights/best.pt         # Trained YOLOv8 pose model
│   └── result.png              # Train/Validation model results
├── model.py                    # Model generation
├── test.py                     # Model testing
├── data.yaml                   # Dataset config used for training
├── README.md                   # This file
