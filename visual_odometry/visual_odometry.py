import cv2
import numpy as np
from .monovideoodometry import MonoVideoOdometry
from .parameters import *


def visual_odometry(image_path="./input/sequences/10/image_0/", pose_path="./input/poses/10.txt"):
    """
    Plots the estimated odometry path using either five point estimation or eight point estimation
    :param image_path: Path to the directory of camera images
    :param pose_path: Path to the directory of pose file
    :return: None
    """

    LUCAS_KANADE_PARAMS = dict(winSize=(21, 21), criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

    vo = MonoVideoOdometry(image_path, pose_path, FOCAL, PP, LUCAS_KANADE_PARAMS)
    trajectory = np.zeros(shape=(800, 1200, 3))

    frame_count = 0
    while vo.hasNextFrame():
        frame_count += 1
        frame = vo.current_frame

        cv2.imshow("Frame", frame)
        k = cv2.waitKey(1)
        if k == 27:
            break

        vo.process_frame()

        estimated_coordinates = vo.get_mono_coordinates()
        true_coordinates = vo.get_true_coordinates()

        print("MSE Error: ", np.linalg.norm(estimated_coordinates - true_coordinates))
        print("x: {}, y: {}, z: {}".format(*[str(pt) for pt in estimated_coordinates]))
        print("True_x: {}, True_y: {}, True_z: {}".format(*[str(pt) for pt in true_coordinates]))

        draw_x, draw_y, draw_z = [int(round(x)) for x in estimated_coordinates]
        true_x, true_y, true_z = [int(round(x)) for x in true_coordinates]

        trajectory = cv2.circle(trajectory, (true_x + 400, true_z + 100), 1, list((0, 0, 255)), 4)
        trajectory = cv2.circle(trajectory, (draw_x + 400, draw_z + 100), 1, list((0, 255, 0)), 4)

        cv2.putText(trajectory, "Actual Position:", (140, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(trajectory, "Red", (270, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(trajectory, "Estimated Odometry Position:", (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(trajectory, "Green", (270, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.imshow("trajectory", trajectory)

        if frame_count % 5 == 0:
            cv2.imwrite(f"./results/trajectory/trajectory_{frame_count}.png", trajectory)

    cv2.imwrite(f"./results/trajectory.png", trajectory)
    cv2.destroyAllWindows()
