import cv2
import numpy as np
import torch

from viz_utils import denormalization, upsampling, save_viz


def viz_attr(
    realimg:np.array,
    unit:torch.tensor,
    ratio = (0.4, 0.6),
):
    '''
    Function for visualizing a neuron capturing an attribute over a given original image.

    @Input:

        realimg[np.array]: original image;

        unit[torch.tensor]: activation map of the unit;

        ratio[tuple]: weights (sum to 1) over heatmap and original image, default as (0.4, 0.6);
    '''
    # upsample feature map to res with the same size as original image
    res, mask = upsampling(
        unit = unit, 
        real = realimg, 
        sampling_type = 'bicubic',
    )

    # # preprocess original image
    # # uncomment this if loaded image is in type torch.tensor 
    # realimg = realimg.numpy()

    # comment this if the original image is not loaded from opencv
    # because only opencv order the three channels as B, R, G
    # realimg = realimg.reshape((224, 224, 3), order='A')
    realimg = np.transpose(realimg, (1, 2, 0))

    realimg = denormalization(realimg)
    # realimg = cv2.cvtColor(realimg, cv2.COLOR_BGR2RGB)

    # generate heatmap
    # res = res.reshape(224, 224, 1)
    res = np.uint8(255-denormalization(res))
    res = cv2.applyColorMap(res, cv2.COLORMAP_JET)
    print(res.shape)
    # apply heatmap to original image
    res = realimg * ratio[1] + res * ratio[0]
    print(res.shape)
    # generate binary mask
    mask = np.uint8(mask * 255)
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    # mask = cv2.applyColorMap(mask, cv2.COLORMAP_JET)

    # apply binary mask to original image
    mask = mask * ratio[0] + realimg * ratio[1]
    return res, mask, realimg
