import random
import numpy as np
from .eight_point import eight_point_estimation
from .parameters import *


def evaluate(previous_frame_points, current_frame_points, fundamental_matrix, error_threshold):
    """
    Evaluate a fundamental matrix over all points
    :param previous_frame_points: Previous frame points
    :param current_frame_points: Current frame points
    :param fundamental_matrix: Fundamental Matrix
    :param error_threshold: Error threshold
    :return: Inliers in previous frame, inliers in current frame, inlier count
    """

    inliers_count = 0
    old_frame_inliers = []
    new_frame_inliers = []

    for i in range(len(previous_frame_points)):
        numerator = (current_frame_points[i] @ fundamental_matrix @ previous_frame_points[i].T) ** 2

        first_epiline = fundamental_matrix @ previous_frame_points[i].T
        second_epiline = np.transpose(fundamental_matrix) @ current_frame_points[i].T
        denominator = first_epiline[0] ** 2 + first_epiline[1] ** 2 + second_epiline[0] ** 2 + second_epiline[1] ** 2

        error = numerator / denominator

        if error < error_threshold:
            old_frame_inliers.append(previous_frame_points[i])
            new_frame_inliers.append(current_frame_points[i])
            inliers_count += 1

    return (
        np.ascontiguousarray(old_frame_inliers),
        np.ascontiguousarray(new_frame_inliers),
        inliers_count,
    )


def RANSAC(previous_frame_points, current_frame_points):
    """
    Selects the best fundamental matrix with RANSAC outlier rejection
    :param previous_frame_points: Previous frame points
    :param current_frame_points: Current frame points
    :return: Fundamental matrix, inlier points, outlier points
    """

    iterations = 0

    previous_frame_inlier_points = None
    current_frame_inlier_points = None
    best_fundamental_matrix = None

    best_inlier_count = 0
    num_points = min(len(previous_frame_points), len(current_frame_points))

    # Iteratively find the best fitting essential matrix and store the inlier points.
    while iterations < NUM_ITERATIONS:
        # To account for noisy data we consider more point than eight
        rand_choice = np.unique([random.randint(0, num_points - 1) for _ in range(25)])
        previous_frame_points_input = np.array([previous_frame_points[x] for x in rand_choice])
        current_frame_points_input = np.array([current_frame_points[x] for x in rand_choice])

        # Find a fundamental matrix for a random subset of points and evaluate the performance
        estimated_fundamental_matrix = eight_point_estimation(previous_frame_points_input, current_frame_points_input)
        old_frame_inliers, new_frame_inliers, current_inlier_count = evaluate(
            previous_frame_points,
            current_frame_points,
            estimated_fundamental_matrix,
            THRESHOLD,
        )

        if current_inlier_count > best_inlier_count:
            best_inlier_count = current_inlier_count
            best_fundamental_matrix = estimated_fundamental_matrix
            previous_frame_inlier_points = old_frame_inliers
            current_frame_inlier_points = new_frame_inliers

        iterations += 1

    return (
        best_fundamental_matrix,
        previous_frame_inlier_points,
        current_frame_inlier_points,
    )
