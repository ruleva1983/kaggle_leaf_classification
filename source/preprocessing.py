import cv2
import numpy as np


def resize_image_to_square(image, side, interpolation=cv2.INTER_NEAREST):
    """
    This function resizes a given image to a square image maintaining the proportionality
    between the two axis.
    :param image: The image to resize
    :type image: Bidimensional numpy.ndarray
    :param side: The side of squared final image
    :type side: Integer
    :param interpolation: The interpolation scheme
    :type: cv2 library interpolation type
    :return: The resized image in numpy.ndarray format
    """
    shape = np.array(image.shape)
    if np.argmax(shape) == 0:
        reshape_shape = (side * shape[1] / shape[0], side)
    else:
        reshape_shape = (side, side * shape[0] / shape[1])
    resized = cv2.resize(image, reshape_shape, interpolation=interpolation)

    x = np.zeros((side, side))
    if np.argmax(shape) == 0:
        init = (reshape_shape[1] - reshape_shape[0]) / 2
        x[:, init:init + reshape_shape[0]] = resized
    else:
        init = (reshape_shape[0] - reshape_shape[1]) / 2
        x[init:init + reshape_shape[1], :] = resized
    return x


def apply_rotation(image, angle):
    """
    Apply a rotation around the center to  the input image. Used for data augmentation
    :param image: The image to rotate
    :type image: Bidimensional numpy.ndarray
    :param angle: The angle of rotation.
    :type angle: floating point
    :return: The rotated image in numpy.nd.array format
    """
    rows, cols = image.shape
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    return cv2.warpAffine(image, M, (cols, rows))


def apply_translation(image, horizontal_shift, vertical_shift):
    """
    Applies a translation to the image taking care that the important parts
    are not cut out.
    :param image: The image to translate
    :type image: Bidimensional numpy.ndarray
    :param horizontal_shift: The horizontal shift measured in number of pixels
    :type horizontal_shift: Integer
    :param vertical_shift: The vertical shift measured in number of pixels
    :type vertical_shift: Integer
    :return: The translated image in numpy.nd.array format
    """
    rows, cols = image.shape

    def _check_validity(image):
        pass

    horizontal_shift, vertical_shift = _check_validity()

    M = np.float32([[1, 0, horizontal_shift], [0, 1, vertical_shift]])
    return cv2.warpAffine(image, M, (cols, rows))