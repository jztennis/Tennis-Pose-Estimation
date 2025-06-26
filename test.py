import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO("runs/pose/tennis_pose_model3/weights/best.pt")  # load trained model
model.info()

# Open video with OpenCV
cap = cv2.VideoCapture("ex1.mp4")

# Desired resolution
res_w, res_h = 1080, 720
target_aspect = res_w / res_h

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # frame = cv2.resize(frame, (1080, 720))
    result = model.predict(source=frame, conf=0.012, stream=False, verbose=False)[0]

    print(f"Detected {len(result.keypoints.xy)} poses this frame")

    for kp in result.keypoints.xy:  # [N, num_keypoints, 2]
        for x, y in kp:
            cv2.circle(frame, (int(x), int(y)), radius=4, color=(0, 255, 0), thickness=-1)

    # # --- Display ---
    cv2.imshow("Pose Estimation", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
