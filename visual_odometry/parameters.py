import cv2
import numpy as np


NUM_ITERATIONS = 25
INFINITY = float("inf")
MIN_INLIER_POINTS = 10
THRESHOLD = 0.1

FOCAL = 718.8560
PP = (607.1928, 185.2157)
IMAGE_SHAPE = (370, 1226)
LUCAS_KANADE_PARAMS = dict(
    winSize=(21, 21),
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
)
DETECTOR = cv2.FastFeatureDetector_create(threshold=25, nonmaxSuppression=True)

K = np.array(
    [
        [7.215377000000e02, 0.000000000000e00, 6.095593000000e02],
        [0.000000000000e00, 7.215377000000e02, 1.728540000000e02],
        [0.000000000000e00, 0.000000000000e00, 1.000000000000e00],
    ]
)
