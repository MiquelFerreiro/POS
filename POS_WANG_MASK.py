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

class ROIS: 

        OJO_IZQ = [26, 22, 23, 24, 110, 25, 130, 247, 30, 29, 27, 28, 56, 190, 243, 112]

        OJO_DER = [253, 254, 339, 255, 359, 467, 260, 259, 257, 258, 286, 414, 463, 341, 256, 252]

        BOCA = [57, 186, 92, 165, 167, 164, 393, 391, 322, 410, 287, 273, 335, 406, 313, 18, 83, 182, 106, 43]

        FULL_FACE = [175, 171, 140, 170, 169, 135, 138, 215, 177, 137, 227, 34, 139, 71, 54, 103, 67, 109, 10, 338, 297, 332, 284,
         301, 368, 264, 447, 366, 401, 435, 367, 364, 394, 395, 369, 396]

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

    mask_ojos = cv2.bitwise_or(
        generateMaskFromPoints(ROIS.OJO_IZQ, face, w, h),
        generateMaskFromPoints(ROIS.OJO_DER, face, w, h),
    )

    mask_excluded = cv2.bitwise_or(mask_ojos, generateMaskFromPoints(ROIS.BOCA, face, w, h))

    mask_full = generateMaskFromPoints(ROIS.FULL_FACE, face, w, h)

    mask_excluded_inv = cv2.bitwise_not(mask_excluded)

    final_mask = cv2.bitwise_and(mask_full, mask_excluded_inv)

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


