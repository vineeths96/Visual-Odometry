import os
from .eight_point import eight_point_estimation_builtin, unscale_fundamental_matrix
from .RANSAC import RANSAC
from .utils import (
    cartesian_to_homogeneous,
    homogeneous_to_cartesian,
    scale_coordinates,
    unscale_coordinates,
)
from .parameters import *


class MonoVideoOdometry(object):
    def __init__(
        self,
        image_file_path,
        pose_file_path,
        focal_length=FOCAL,
        pp=PP,
        K=K,
        lk_params=LUCAS_KANADE_PARAMS,
        fivepoint=False,
        image_shape=IMAGE_SHAPE,
        detector=DETECTOR,
    ):
        """
        Initializes the class
        :param img_file_path: File path that leads to image sequences
        :param pose_file_path: File path that leads to true poses from image sequence
        :param focal_length: Focal length of camera used in image sequence
        :param pp: Principal point of camera in image sequence
        :param lk_params: Parameters for Lucas Kanade optical flow
        :param fivepoint: Five point or eight point algorithm
        :param image_shape: Shape of image frames
        :param detector: Most types of OpenCV feature detectors
        """

        self.file_path = image_file_path
        self.detector = detector
        self.focal = focal_length
        self.pp = pp
        self.K = K
        self.lk_params = lk_params
        self.image_shape = image_shape
        self.fivepoint = fivepoint

        self.R = np.zeros(shape=(3, 3))
        self.t = np.zeros(shape=(3, 3))

        self.id = 0
        self.n_features = 0

        try:
            if not all([".png" in x for x in os.listdir(image_file_path)]):
                raise ValueError("image_file_path is not correct and does not have exclusively png files")
        except Exception as e:
            print(e)
            raise ValueError("The designated image_file_path does not exist")

        try:
            with open(pose_file_path) as f:
                self.pose = f.readlines()
        except Exception as e:
            print(e)
            raise ValueError("The pose_file_path is not valid or did not lead to a txt file")

        self.process_frame()

    def hasNextFrame(self):
        """
        Determine whether there are remaining frames in the folder to process
        :return: Whether there are still frames in the folder to process
        """

        return self.id < len(os.listdir(self.file_path))

    def detect(self, img):
        """
        Detect features and parse into useable format
        :param img: Image for which to detect keypoints on
        :return: A sequence of points in (x, y) coordinate format denoting location of detected keypoint
        """

        p0 = self.detector.detect(img)

        return np.array([x.pt for x in p0], dtype=np.float32).reshape(-1, 1, 2)

    def visual_odometry(self):
        """
        Estimates the location using visual odometery. Depending on self.fivepoint flag either a five point
        method or eight point method is used. If features fall out of frame such that there are less than 2000
        features remaining, a new feature detection is triggered.
        """

        if self.n_features < 2000:
            self.p0 = self.detect(self.old_frame)

        # Calculate optical flow between frames, and track matched points from frame to frame
        self.p1, st, err = cv2.calcOpticalFlowPyrLK(
            self.old_frame, self.current_frame, self.p0, None, **self.lk_params
        )

        # Save the good points from the optical flow
        self.previous_frame_points = self.p0[st == 1]
        self.current_frame_points = self.p1[st == 1]

        if self.fivepoint:
            if self.id < 2:
                E, _ = cv2.findEssentialMat(
                    self.current_frame_points,
                    self.previous_frame_points,
                    self.focal,
                    self.pp,
                    cv2.RANSAC,
                    0.999,
                    1.0,
                    None,
                )

                _, self.R, self.t, _ = cv2.recoverPose(
                    E,
                    self.previous_frame_points,
                    self.current_frame_points,
                    self.R.copy(),
                    self.t,
                    self.focal,
                    self.pp,
                    None,
                )
            else:
                E, _ = cv2.findEssentialMat(
                    self.current_frame_points,
                    self.previous_frame_points,
                    self.focal,
                    self.pp,
                    cv2.RANSAC,
                    0.999,
                    1.0,
                    None,
                )

                _, R, t, _ = cv2.recoverPose(
                    E,
                    self.previous_frame_points,
                    self.current_frame_points,
                    self.R.copy(),
                    self.t.copy(),
                    self.focal,
                    self.pp,
                    None,
                )

                absolute_scale = self.get_absolute_scale()
                if absolute_scale > 0.1 and abs(t[2][0]) > abs(t[0][0]) and abs(t[2][0]) > abs(t[1][0]):
                    self.t = self.t + absolute_scale * self.R.dot(t)
                    self.R = R.dot(self.R)

            # Save the total number of good features
            self.n_features = self.current_frame_points.shape[0]
        else:
            # Fundamental matrix inbuilt estimation using eight point inbuilt method
            """
            fundamental_matrix_inbuilt, inliers = eight_point_estimation_builtin(
                self.previous_frame_points, self.current_frame_points
            )
            previous_frame_inliers = self.previous_frame_points[inliers.ravel() == 1]
            current_frame_inliers = self.current_frame_points[inliers.ravel() == 1]
            fundamental_matrix = fundamental_matrix_inbuilt.T
            """

            # Fundamental matrix inbuilt estimation using eight point custom method
            M = max(self.image_shape)
            self.previous_frame_points = scale_coordinates(self.previous_frame_points, M)
            self.current_frame_points = scale_coordinates(self.current_frame_points, M)

            self.previous_frame_points = cartesian_to_homogeneous(self.previous_frame_points)
            self.current_frame_points = cartesian_to_homogeneous(self.current_frame_points)

            fundamental_matrix, previous_frame_inliers, current_frame_inliers = RANSAC(
                self.previous_frame_points, self.current_frame_points
            )
            # fundamental_matrix, previous_frame_inliers, current_frame_inliers = RANSAC(
            #     self.previous_frame_points[inliers.ravel() == 1], self.current_frame_points[inliers.ravel() == 1]
            # )
            fundamental_matrix = unscale_fundamental_matrix(fundamental_matrix, M)

            previous_frame_inliers = homogeneous_to_cartesian(previous_frame_inliers)
            current_frame_inliers = homogeneous_to_cartesian(current_frame_inliers)

            previous_frame_inliers = unscale_coordinates(previous_frame_inliers, M)
            current_frame_inliers = unscale_coordinates(current_frame_inliers, M)

            E = self.K.T @ fundamental_matrix @ self.K

            # Initialize t and R for first two frames, update it successively
            if self.id < 2:
                _, self.R, self.t, _ = cv2.recoverPose(
                    E,
                    previous_frame_inliers,
                    current_frame_inliers,
                    self.R.copy(),
                    self.t,
                    self.focal,
                    self.pp,
                    None,
                )
            else:
                _, R, t, _ = cv2.recoverPose(
                    E,
                    previous_frame_inliers,
                    current_frame_inliers,
                    self.R.copy(),
                    self.t.copy(),
                    self.focal,
                    self.pp,
                    None,
                )

                absolute_scale = self.get_absolute_scale()
                if absolute_scale > 0.1 and abs(t[2][0]) > abs(t[0][0]) and abs(t[2][0]) > abs(t[1][0]):
                    self.t = self.t + absolute_scale * self.R.dot(t)
                    self.R = R.dot(self.R)

        # Save the total number of good features
        self.n_features = self.current_frame_points.shape[0]

    def get_mono_coordinates(self):
        """
        Multiply by the diagonal matrix to fix our vector onto same coordinate axis as true values
        :return: Array in format [x, y, z]
        """

        diag = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, -1]])
        adj_coord = np.matmul(diag, self.t)

        return adj_coord.flatten()

    def get_true_coordinates(self):
        """
        Returns true coordinates of vehicle
        :return: Array in format [x, y, z]
        """

        return self.true_coord.flatten()

    def get_absolute_scale(self):
        """
        Estimation of scale for multiplying translation vectors
        :return: Scalar multiplier
        """

        pose = self.pose[self.id - 1].strip().split()
        x_prev = float(pose[3])
        y_prev = float(pose[7])
        z_prev = float(pose[11])
        pose = self.pose[self.id].strip().split()
        x = float(pose[3])
        y = float(pose[7])
        z = float(pose[11])

        true_vect = np.array([[x], [y], [z]])
        self.true_coord = true_vect
        prev_vect = np.array([[x_prev], [y_prev], [z_prev]])

        return np.linalg.norm(true_vect - prev_vect)

    def process_frame(self):
        """
        Processes images in sequence frame by frame
        """

        if self.id < 2:
            self.old_frame = cv2.imread(self.file_path + str().zfill(6) + ".png", 0)
            self.current_frame = cv2.imread(self.file_path + str(1).zfill(6) + ".png", 0)
            self.visual_odometry()
            self.id = 2
        else:
            self.old_frame = self.current_frame
            self.current_frame = cv2.imread(self.file_path + str(self.id).zfill(6) + ".png", 0)
            self.visual_odometry()
            self.id += 1
