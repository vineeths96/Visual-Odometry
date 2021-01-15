import random
import numpy as np
from eight_point import eight_point_estimation


def evaluate(src, dst, fundamental_matrix, error_threshold):
    inliers_count = 0
    cumulative_error = 0
    old_frame_inliers = []
    new_frame_inliers = []

    for i in range(len(src)):
        numerator = (dst[i] @ fundamental_matrix @ src[i].T) ** 2

        first_epiline = fundamental_matrix @ src[i].T
        second_epiline = np.transpose(fundamental_matrix) @ dst[i].T
        denominator = first_epiline[0] ** 2 + first_epiline[1] ** 2 + second_epiline[0] ** 2 + second_epiline[1] ** 2

        error = numerator / denominator

        if error < error_threshold:
            cumulative_error += error
            old_frame_inliers.append(src[i])
            new_frame_inliers.append(dst[i])
            inliers_count += 1

    avg_error = cumulative_error / inliers_count

    return np.ascontiguousarray(old_frame_inliers), np.ascontiguousarray(new_frame_inliers), avg_error


def RANSAC(src, dst):
    num_iterations = 10
    iterations = 0

    old_frame_points = None
    new_frame_points = None
    best_fundamental_matrix = None

    error = float('inf')
    num_points = min(len(src), len(dst))

    # Adaptively determining the number of iterations
    while iterations < num_iterations:
        rand_choice = np.unique([random.randint(0, num_points - 1) for _ in range(25)])
        src_input = np.array([src[x] for x in rand_choice])
        dst_input = np.array([dst[x] for x in rand_choice])

        estimated_fundamental_matrix = eight_point_estimation(src_input, dst_input)
        # from utils import unscale_fundamental_matrix
        # estimated_fundamental_matrix = unscale_fundamental_matrix(estimated_fundamental_matrix, 1726)

        # count the inliers within the threshold
        old_frame_inliers, new_frame_inliers, current_error = evaluate(src, dst, estimated_fundamental_matrix, error)

        # check for the best model
        if current_error < error:
            best_fundamental_matrix = estimated_fundamental_matrix
            old_frame_points = old_frame_inliers
            new_frame_points = new_frame_inliers

        iterations += 1

    return best_fundamental_matrix, old_frame_points, new_frame_points



# def RANSAC_LOOP1(src, dst):
    # RANSAC_TIMES = 10
    # num_vals = len(src)
    # minval = 9999999
    # F_hold = None
    # for i in range(RANSAC_TIMES):
    #     # print(i)
    #     # Choose random 8.
    #     rand_choice = [random.randint(0, num_vals-1) for x in range(25)]
    #     src_input = np.array([src[x] for x in rand_choice])
    #     dst_input = np.array([dst[x] for x in rand_choice])
    #
    #     F_temp = eight_point_estimation(src_input, dst_input)
    #
    #     from utils import unscale_fundamental_matrix
    #     F_temp = unscale_fundamental_matrix(F_temp, 1726)
    #
    #     # print(F_temp)
    #     tot = 0
    #     for j in range(num_vals):
    #         vec1 = np.transpose(np.array([dst[j][0], dst[j][1], 1]))
    #         vec2 = np.array([src[j][0], src[j][1], 1])
    #         temp2 = np.matmul(vec1, F_temp)
    #         tot += np.matmul(temp2, vec2)
    #     if tot < 0:
    #         tot = -1 * tot
    #     if tot < minval:
    #         F_hold = F_temp
    #         minval = tot
    #
    # print(minval)
    # return F_hold
