import cv2
import glob
import numpy as np
import os.path as osp
from argparse import ArgumentParser
from utils.compute_iou import compute_ious


def segment_fish(img):
    """
    This method should compute masks for given image
    Params:
        img (np.ndarray): input image in BGR format
    Returns:
        mask (np.ndarray): fish mask. should contain bool values
    """
    img = img[:,:,::-1]
    img_HSV = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    lower_red = np.array([1, 190, 150])
    upper_red = np.array([30, 255, 255])
    mask_red = cv2.inRange(img_HSV, lower_red, upper_red)
    lower_white = np.array([60, 0, 200])
    upper_white = np.array([145, 150, 255])
    mask_white = cv2.inRange(img_HSV, lower_white, upper_white)
    mask = mask_red | mask_white
    kernel = np.ones((4, 4),np.uint8)
    erosion = cv2.erode(mask, kernel, iterations = 5)
    dilation = cv2.dilate(erosion, kernel, iterations = 10)
    scale_percent = 50 # percent of original size
    width = int(mask.shape[1] * scale_percent / 100)
    height = int(mask.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv2.resize(mask, dim, interpolation = cv2.INTER_AREA)
    kernel = np.ones((4, 4),np.uint8)
    erosion = cv2.erode(resized, kernel, iterations = 3)
    dilation = cv2.dilate(erosion, kernel, iterations = 8)
    dim = (mask.shape[1], mask.shape[0])
    resized = cv2.resize(dilation, dim, interpolation = cv2.INTER_AREA)
    ret, thresh1 = cv2.threshold(resized,127,255,cv2.THRESH_BINARY)

    return thresh1


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--is_train", action="store_true")
    args = parser.parse_args()
    stage = 'train' if args.is_train else 'test'

    data_root = osp.join("dataset", stage, "imgs")
    img_paths = glob.glob(osp.join(data_root, "*.jpg"))
    len(img_paths)

    masks = dict()
    for path in img_paths:
        img = cv2.imread(path)
        mask = segment_fish(img)
        masks[osp.basename(path)] = mask
    print(compute_ious(masks, osp.join("dataset", stage, "masks")))
