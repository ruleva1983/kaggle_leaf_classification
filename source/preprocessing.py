import cv2
import numpy as np
from copy import deepcopy
import pandas as pd


def augment_data_set(files, features, nb_augment, rot_prob, tr_prob, seed=0, IMAGE_DIM=200, shuffle=False):
    """
    Augmentation function for dataset.
    :param files: List of image files
    :type files: iterable
    :param features: A dataframe with the external features and labels
    :type features: pandas.DataFrame
    :param nb_augment: Number of augmented copies per image
    :type nb_augment: Integer
    :param rot_prob: Probability of applying a rotation
    :type rot_prob: float
    :param tr_prob: Probability of applying a translation
    :type tr_prob: float
    :param seed: Random number generator seed
    :type seed: Integer
    :param IMAGE_DIM: The dimension of squared images
    :type IMAGE_DIM: Integer
    :return: A numpy array containing the images (n_images, x_dim, y_dim, 1), A numpy array containing the corresponing
             features (n_images, n_features), A numpy array containing the labels in one hot encoding.
    """
    np.random.seed=seed
    X_images = []
    X_features = []
    y = []
    for f in files:
        id_ = int(f[:-4])
        try:
            feat = features.drop('species', 1).loc[[id_]].as_matrix()
        except KeyError:
            continue
        label = features.get_value(id_, "species")

        image = cv2.imread(f, 0)
        image = resize_image_to_square(image, IMAGE_DIM)
        for n in range(nb_augment):
            augmented = deepcopy(image)
            if np.random.uniform(0, 1) < rot_prob:
                alpha = np.random.uniform(0,360)
                augmented = apply_rotation(image, angle=alpha)
            if np.random.uniform(0, 1) < tr_prob:
                x_shift = np.random.randint(-5, 5)
                y_shift = np.random.randint(-5, 5)
                augmented = apply_translation(augmented, horizontal_shift=x_shift, vertical_shift=y_shift)
            X_images.append(augmented)
            X_features.append(feat.T)
            y.append(label)
    X_images = np.array(X_images)
    X_features = np.array(X_features)

    X_images = X_images.reshape((-1, X_images.shape[1], X_images.shape[2], 1))
    X_features = X_features.reshape((-1, X_features.shape[1]))
    y = pd.get_dummies(y).as_matrix().astype(np.int64)

    if shuffle:
        p = np.random.permutation(len(X_images))
        X_images = X_images[p]
        X_features = X_features[p]
        y = y[p]

    return X_images, X_features, y




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

    #def _check_validity(image):
    #    pass

    #horizontal_shift, vertical_shift = _check_validity()

    M = np.float32([[1, 0, horizontal_shift], [0, 1, vertical_shift]])
    return cv2.warpAffine(image, M, (cols, rows))