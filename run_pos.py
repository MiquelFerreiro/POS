import cv2
import numpy as np
from POS_WANG import POS_WANG
from POS_WANG_MASK import POS_WANG_MASK
import utils

# Video a procesar
#video_path = "vid.avi"
#video_path = "vid_crop.avi"
video_path = "Dataset/Patient_1/Q1_1/vid_crop.avi"


# Cargar video y convertir a RGB
cap = cv2.VideoCapture(video_path)
frames = []
while True:
    ret, frame = cap.read()
    if not ret:
        break
    #frame = cv2.resize(frame, (128, 128))  # resize opcional
    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
cap.release()

frames = np.array(frames)
print(f"Loaded {len(frames)} frames from video.")

# Inferir señal rPPG
fs = 60  # fps del video
#rppg_signal = POS_WANG(frames, fs)
try:
    rppg_signal = POS_WANG_MASK(frames, fs)
    np.save("pos_output.npy", rppg_signal)
    print("Saved rPPG signal to pos_output.npy")
except Exception as e:
    print("Error during POS_WANG_MASK:", e)

