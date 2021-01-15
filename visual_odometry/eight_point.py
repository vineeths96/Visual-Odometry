import cv2
import numpy as np


def unscale_fundamental_matrix(fundamental_matrix, M):
    """
    Unscale fundamental matrix by coordinate scaling factor
    :param fundamental_matrix:
    :param M: Scaling factor
    :return: Unscaled fundamental matrix
    """

    T = np.diag([1 / M, 1 / M, 1])
    unscaled_F = T.T.dot(fundamental_matrix).dot(T)

    return unscaled_F


def eight_point_estimation_builtin(previous_frame_points_points, current_frame_points):
    """
    Open CV Eight point estimation of fundamental matrix
    :param previous_frame_points_points: Previous frame matched key points
    :param current_frame_points: Current frame matched key points
    :return: Fundamental matrix, inlier points
    """

    _, inliers = cv2.findFundamentalMat(previous_frame_points_points, current_frame_points, cv2.FM_RANSAC)
    F, _ = cv2.findFundamentalMat(
        previous_frame_points_points[inliers.ravel() == 1],
        current_frame_points[inliers.ravel() == 1],
        cv2.FM_8POINT,
    )

    return F, inliers


def eight_point_estimation(previous_frame_points_points, current_frame_points):
    """
    Eight point estimation of fundamental matrix
    :param previous_frame_points_points: Previous frame matched key points
    :param current_frame_points: Current frame matched key points
    :return: Fundamental matrix, inlier points
    """

    A = np.zeros((previous_frame_points_points.shape[0], 9))

    for i in range(previous_frame_points_points.shape[0]):
        A[i, :] = np.kron(previous_frame_points_points[i, :], current_frame_points[i, :])

    # Pick the smallest eigenvector
    u, s, vh = np.linalg.svd(A)
    v = vh.T
    f = v[:, -1].reshape(3, 3)

    # Force essential matrix constraint
    U, S, V = np.linalg.svd(f)
    F = U.dot(np.diag([1, 1, 0]).dot(V))

    return F
