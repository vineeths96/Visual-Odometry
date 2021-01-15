import numpy as np


def cartesian_to_homogeneous(cartesian_point):
    homogeneous_point = np.ones([cartesian_point.shape[0], cartesian_point.shape[1] + 1])
    homogeneous_point[:, :-1] = cartesian_point

    return homogeneous_point


def homogeneous_to_cartesian(homogeneous_point):
    return homogeneous_point[:, :-1]


def scale_coordinates(point, M):
    return point / M


def unscale_coordinates(point, M):
    return point * M


def unscale_fundamental_matrix(fundamental_matrix, M):
    T = np.diag([1 / M, 1 / M, 1])
    unscaled_F = T.T.dot(fundamental_matrix).dot(T)

    return unscaled_F
