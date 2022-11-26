import itertools
import os
import time
from collections import Counter

# coding=utf-8
from typing import Union
import cv2
import numpy as np
from numba import jit


def calcEntropyRGB2d(img: np.array, size: Union[np.array, list, set, tuple]):
    '''
    Function calcEntropyRGB2d.

    For calculating 2d Shannon Information Entropy for an RGB image.

    @Args:

        img [numpy.array]: [H * W * C];

        size [tuple]: (H, W);
    '''
    img = cv2.resize(img, size)
    img = cv2.normalize(
        src = img,
        dst = None,
        alpha = 0,
        beta = 1,
        norm_type = cv2.NORM_MINMAX,
        dtype = cv2.CV_32F,
    )
    
    H1 = __calcEntropy2d(img[:, :, 0], 3, 3)
    H2 = __calcEntropy2d(img[:, :, 1], 3, 3)
    H3 = __calcEntropy2d(img[:, :, 2], 3, 3)
    return (H1 + H2 + H3) / 3


def __calcEntropy2d(img, win_w=3, win_h=3):
    '''
    Function calcEntropy2d.

    For calculating 2d Shannon Information Entropy for an gray-scaled image.
    '''
    height = img.shape[0]

    ext_x = int(win_w / 2)
    ext_y = int(win_h / 2)

    ext_h_part = np.zeros([height, ext_x], img.dtype)
    tem_img = np.hstack((ext_h_part, img, ext_h_part))
    ext_v_part = np.zeros([ext_y, tem_img.shape[1]], img.dtype)
    final_img = np.vstack((ext_v_part, tem_img, ext_v_part))

    new_width = final_img.shape[1]
    new_height = final_img.shape[0]

    # iterater over all bi-tuples
    IJ = [
        __calcIJ(final_img[j - ext_y:j + ext_y + 1, i - ext_x:i + ext_x + 1])    
        for [i, j] 
        in itertools.product(*[range(ext_x, new_width - ext_x), range(ext_y, new_height - ext_y)])
    ]
    
    Fij = Counter(IJ).items()

    # calculate frequencies over bi-tuples
    Pij = [
        (item[1] * 1.0 / (new_height * new_width))
        for item
        in Fij
    ]

    H_tem = [
        (-item * (np.log(item)) / np.log(2))
        for item 
        in Pij
    ]

    return np.sum(H_tem)


@jit(nopython=True)
def __calcIJ(img_patch):
    total_p = img_patch.shape[0] * img_patch.shape[1]
    if total_p % 2 != 0:
        center_p = img_patch[int(img_patch.shape[0] / 2), int(img_patch.shape[1] / 2)]
        mean_p = (np.sum(img_patch) - center_p) / (total_p - 1)
        return (center_p, mean_p)
    else:
        pass

