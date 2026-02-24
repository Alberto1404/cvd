import numpy as np

def normalize(image: np.ndarray) -> np.ndarray:
    """
    Normalize image from [0, 255] (uint8) to [0, 1] (float64).

    :param image: Input image as a numpy array of type uint8.
    :type image: np.ndarray
    :return: Normalized image as a numpy array of type float64.
    :rtype: np.ndarray
    """

    return image.astype(np.float64) / 255.0

def denormalize(image: np.ndarray) -> np.ndarray:
    """
    Denormalize image from [0, 1] (float64) to [0, 255] (uint8).

    :param image: Input image as a numpy array of type float64.
    :type image: np.ndarray
    :return: Denormalized image as a numpy array of type uint8.
    :rtype: np.ndarray
    """

    return (np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
