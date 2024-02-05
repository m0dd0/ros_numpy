import numpy as np
from sensor_msgs.msg import CameraInfo

from .registry import converts_from_numpy, converts_to_numpy


@converts_from_numpy(CameraInfo)
def cam_intrinsics_to_ros_msg(
    cam_intrinsics: np.ndarray,
    height: int = 480,
    width: int = 640,
    distortion_model: str = "plumb_bob",
    D: np.ndarray = np.zeros(5),
    R: np.ndarray = np.eye(3, dtype=float),
    P: np.ndarray = None,
) -> CameraInfo:
    """Convert camera intrinsics to a ROS CameraInfo message.

    Args:
        cam_intrinsics: Camera intrinsics matrix (3x3)
        height: Image height (default: 480)
        width: Image width (default: 640)
        distortion_model: Distortion model (default: "plumb_bob")
        D: Distortion coefficients (default: np.zeros(5))
        R: Rectification matrix (default: np.eye(3, dtype=float))
        P: Projection matrix (default: same as cam_intrinsics)

    Returns:
        CameraInfo message
    """

    # http://docs.ros.org/en/melodic/api/sensor_msgs/html/msg/CameraInfo.html
    msg = CameraInfo()

    msg.height = height
    msg.width = width
    msg.distortion_model = distortion_model
    msg.D = D
    msg.K = cam_intrinsics.flatten()
    msg.R = R.flatten()
    if P is None:
        P = np.zeros((3, 4))
        P[0:3, 0:3] = cam_intrinsics
    msg.P = P.flatten()

    return msg


@converts_to_numpy(CameraInfo)
def cam_intrinsics_from_ros_msg(
    msg: CameraInfo,
) -> np.ndarray:
    """Convert a ROS CameraInfo message to camera intrinsics.

    Args:
        msg: CameraInfo message

    Returns:
        Camera intrinsics matrix (3x3)
    """
    return np.array(msg.K).reshape(3, 3)
