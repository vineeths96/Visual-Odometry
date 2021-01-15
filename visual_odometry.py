from visual_odometry.visual_odometry import visual_odometry


# Estimate and plot the odometry
visual_odometry(
    image_path="./input/sequences/03/image_0/",
    pose_path="./input/poses/03.txt",
    fivepoint=False,
)
