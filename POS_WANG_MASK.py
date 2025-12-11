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


# def _process_video(frames):
#     """Calculates the average value of each frame."""
#     RGB = []
#     for frame in frames:
#         summation = np.sum(np.sum(frame, axis=0), axis=0)
#         RGB.append(summation / (frame.shape[0] * frame.shape[1]))
#     return np.asarray(RGB)

def _process_video_mediapipe(frames):
    RGB = []

    face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5)
    
    for frame in frames:
        mask = get_skin_mask(frame, face_mesh)

        skin_pixels = frame[mask > 0]

        if len(skin_pixels) == 0:
            avg = np.mean(frame.reshape(-1,3), axis=0)
        else:
            avg = np.mean(skin_pixels, axis=0)

        RGB.append(avg)

    return np.asarray(RGB)

def generateMaskFromPoints(ROI, face, w, h):
    points = []
    for current_point in ROI:
        x = int(face.landmark[current_point].x * w)
        y = int(face.landmark[current_point].y * h)
        points.append([x, y])

    mask = np.zeros((h, w), dtype=np.uint8)
    points = np.array(points, dtype=np.int32)
    cv2.fillConvexPoly(mask, points, 255)

    return mask

def get_skin_mask(frame, face_mesh):

    h, w, _ = frame.shape

    results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    if not results.multi_face_landmarks:
        print("Error: Sin máscara")
        return np.ones((h, w), dtype=np.uint8)  # fallback: sin mascara

    face = results.multi_face_landmarks[0]

    ROIS = [
        # FRENTE
        #[104, 103, 67, 109, 10, 338, 297, 332, 333, 299, 337, 151, 108, 69]

        #MEJILLA IZQ
        #, [206, 216, 214, 192, 147, 123, 117, 118, 101, 36]
        #, [206, 216, 207, 187, 123, 117, 118, 101, 36]
        #, [206, 205, 50, 118, 101, 36]
        #, [206, 205, 50, 117, 118, 119, 100, 142, 129, 203]

        #MEJILLA DER
        #, [358, 371, 329, 348, 347, 280, 425, 426, 423]

        #FULL FACE
        list(range(468))
        
    ]

    final_mask = np.zeros((h, w), dtype=np.uint8)
    
    for ROI in ROIS:
        current_mask = generateMaskFromPoints(ROI, face, w, h)
        final_mask = cv2.bitwise_or(final_mask, current_mask)

    # suavizar bordes
    final_mask = cv2.GaussianBlur(final_mask, (25, 25), 0)

    return final_mask


def POS_WANG_MASK(frames, fs = 60):
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


