import cv2
import numpy as np


def eight_point_estimation_builtin(good_old, good_new):
    _, inliers = cv2.findFundamentalMat(good_old, good_new, cv2.FM_RANSAC)
    F, _ = cv2.findFundamentalMat(good_old[inliers.ravel() == 1], good_new[inliers.ravel() == 1], cv2.FM_8POINT)

    return F, inliers


def eight_point_estimation(good_old, good_new):
    A = np.zeros((good_old.shape[0], 9))

    for i in range(good_old.shape[0]):
        A[i, :] = np.kron(good_old[i, :], good_new[i, :])

    u, s, vh = np.linalg.svd(A)
    v = vh.T
    f = v[:, -1].reshape(3, 3)

    U, S, V = np.linalg.svd(f)
    S[-1] = 0
    F = U.dot(np.diag([1, 1, 0]).dot(V))

    return F
