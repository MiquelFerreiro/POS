"""POS
Wang, W., den Brinker, A. C., Stuijk, S., & de Haan, G. (2017). 
Algorithmic principles of remote PPG. 
IEEE Transactions on Biomedical Engineering, 64(7), 1479-1491. 
"""

import math

import numpy as np
from scipy import signal
import utils
import mediapipe as mp
import cv2
from scipy.spatial import ConvexHull


mp_face_mesh = mp.solutions.face_mesh

# Índices de puntos relevantes (frente + mejillas)


def _process_video(frames):
    """Calculates the average value of each frame."""
    RGB = []
    for frame in frames:
        summation = np.sum(np.sum(frame, axis=0), axis=0)
        RGB.append(summation / (frame.shape[0] * frame.shape[1]))
    return np.asarray(RGB)

def get_skin_mask_mediapipe(frame):
    h, w, _ = frame.shape
    with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5) as face_mesh:

        results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        if not results.multi_face_landmarks:
            return np.ones((h, w), dtype=np.uint8)  # fallback: sin mascara

        face = results.multi_face_landmarks[0]
        
        # obtener coordenadas de región de piel
        points = []

        ROI_IDX = []
        
        # for idx in ROI_IDX:
        #     x = int(face.landmark[idx].x * w)
        #     y = int(face.landmark[idx].y * h)
        #     points.append([x, y])

        # points = np.array(points, dtype=np.int32)

        mask = np.zeros((h, w), dtype=np.uint8)
        # cv2.fillConvexPoly(mask, points, 255)

        points = np.array([[int(lmk.x * w), int(lmk.y * h)] 
                   for lmk in face.landmark])
        hull = ConvexHull(points)
        hull_points = points[hull.vertices]
        cv2.fillConvexPoly(mask, hull_points, 255)


        # suavizar bordes
        mask = cv2.GaussianBlur(mask, (25, 25), 0)

        return mask


def _process_video_mediapipe(frames):
    RGB = []
    for frame in frames:
        mask = get_skin_mask_mediapipe(frame)

        skin_pixels = frame[mask > 0]

        if len(skin_pixels) == 0:
            avg = np.mean(frame.reshape(-1,3), axis=0)
        else:
            avg = np.mean(skin_pixels, axis=0)

        RGB.append(avg)

    return np.asarray(RGB)


def POS_WANG_MASK(frames, fs):
    WinSec = 1.6
    RGB = _process_video_mediapipe(frames)
    N = RGB.shape[0]
    H = np.zeros((1, N))
    l = math.ceil(WinSec * fs)

    for n in range(N):
        m = n - l
        if m >= 0:
            Cn = np.true_divide(RGB[m:n, :], np.mean(RGB[m:n, :], axis=0))
            Cn = np.asmatrix(Cn).H
            S = np.matmul(np.array([[0, 1, -1], [-2, 1, 1]]), Cn)
            h = S[0, :] + (np.std(S[0, :]) / np.std(S[1, :])) * S[1, :]
            mean_h = np.mean(h)
            for temp in range(h.shape[1]):
                h[0, temp] = h[0, temp] - mean_h
            H[0, m:n] = H[0, m:n] + (h[0])

    BVP = H
    BVP = utils.detrend(np.asmatrix(BVP).H, 100)
    BVP = np.asarray(np.transpose(BVP))[0]
    b, a = signal.butter(1, [0.75 / fs * 2, 3 / fs * 2], btype='bandpass')
    BVP = signal.filtfilt(b, a, BVP.astype(np.double))
    return BVP


