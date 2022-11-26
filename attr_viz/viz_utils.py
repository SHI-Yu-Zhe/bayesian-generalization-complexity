import copy
import os

import cv2
import numpy as np
import torch
from scipy.interpolate import RectBivariateSpline


def upsampling(
    unit:torch.tensor, 
    real:np.array, 
    sampling_type:str,
    thres_gen=5e-2,
):
    '''
    Upsample the activation map of a single unit, exploiting bilinear/bicubic upsampling.

    args:

        unit: activation map of the unit;

        real: original image;

        type[str]: select upsampling algorithm between 'bilinear' and 'bicubic';

        thres_gen[float]: a scalar for generating threshold; 
    '''
    if sampling_type == 'bilinear':
        kx, ky = 1, 1
    elif sampling_type == 'bicubic':
        kx, ky = 3, 3
    else:
        raise Exception('Sampling Type Error! Select one from (bilinear, bicubic).')

    x, y = unit.shape[0], unit.shape[1]
    tx, ty = real.shape[-2], real.shape[-1]

    ax, ay, ix, iy = __coordinate_asnm(x, y, tx, ty)
    
    unit = unit.numpy()
    interp = RectBivariateSpline(
        x = ax, 
        y = ay, 
        z = unit, 
        kx = kx, 
        ky = ky,
    )
    res = interp(ix, iy, grid=True)

    # normalize
    res = (res - np.min(res)) / (np.max(res) - np.min(res))

    # dynamic thresholding and generating neuron masks
    thres, mask = __adaptive_thres(res, thres_gen)
    return res, mask


def __adaptive_thres(res, thres_gen=5e-2):
    '''
    For dynamic thresholding.
    '''
    mask = copy.deepcopy(res)

    # P(f(x) > T) = 0.005
    res = res.reshape(-1)
    res = abs(np.sort(-res))
    res = res[: round(res.shape[0] * thres_gen) + 1]
    thres = res[-1]

    # generating neuron masks
    mask[mask > thres] = 1.
    mask[mask <= thres] = 0.
    return thres, mask


def __coordinate_asnm(x, y, tx, ty):
    if x % 2 == 0:
        # for even number
        ctrx, ctry = tx // 2, ty // 2
        xstep, ystep = tx // x, ty // y
        xmargin, ymargin = xstep // 2, ystep // 2
        xm1, xm2, ym1, ym2 = ctrx - xmargin, ctrx + xmargin, ctry - ymargin, ctry + ymargin

        ax = [x for x in range(xmargin, ctrx, xstep)] + [x for x in range(xm2, tx, xstep)]
        ay = [y for y in range(ymargin, ctry, ystep)] + [y for y in range(ym2, ty, ystep)]

    else:
        # for odd number
        ctrx, ctry = tx // 2, ty // 2
        xstep, ystep = tx // x, ty // y
        xmargin, ymargin = xstep // 2, ystep // 2
        xm1, xm2, ym1, ym2 = ctrx - xmargin, ctrx + xmargin, ctry - ymargin, ctry + ymargin

        ax = [x for x in range(ctrx, 0, - xstep)][:: -1] + [x for x in range(ctrx + xstep, tx, xstep)]
        ay = [y for y in range(ctry, 0, - ystep)][:: -1] + [y for y in range(ctry + ystep, ty, ystep)]

    ix = [x for x in range(tx)]
    iy = [y for y in range(ty)]

    ax, ay, ix, iy = np.array(ax), np.array(ay), np.array(ix), np.array(iy)
    return ax, ay, ix, iy


def denormalization(img):
    '''
    Denormalization of images.

    args:

        img[np.array];
    '''
    return np.uint8(((img - np.min(img)) / (np.max(img) - np.min(img))) * 255)


def save_viz(savepath:str, viz:np.array):
    '''
    Save visualized image.

    @Input:

        savepath[str]: relative save path;

        viz[np.array]: visualization result;
    '''
    cv2.imwrite(savepath, viz)
    return 

