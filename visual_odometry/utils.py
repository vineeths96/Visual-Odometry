import numpy as np


def cartesian_to_homogeneous(cartesian_point):
    """
    Convert Cartesian coordinates to Homogeneous coordinates
    :param cartesian_point: Cartesian coordinates
    :return: Homogeneous coordinates
    """
    homogeneous_point = np.ones(
        [cartesian_point.shape[0], cartesian_point.shape[1] + 1]
    )
    homogeneous_point[:, :-1] = cartesian_point

    return homogeneous_point


def homogeneous_to_cartesian(homogeneous_point):
    """
    Convert Homogeneous coordinates to Cartesian coordinates
    :param homogeneous_point: Homogeneous coordinates
    :return: Cartesian coordinates
    """

    return homogeneous_point[:, :-1]


def scale_coordinates(point, M):
    """
    Scale coordinates by M
    :param point: Point
    :param M: Scaling factor
    :return: Scaled points
    """

    return point / M


def unscale_coordinates(point, M):
    """
    Unscale coordinates by M
    :param point: Point
    :param M: Scaling factor
    :return: Unscaled points
    """

    return point * M
