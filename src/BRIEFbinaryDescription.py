from skimage.feature import (corner_peaks, corner_harris, BRIEF)
from skimage.color import rgb2gray

def BRIEF_skimag(img):
    img1 = rgb2gray(img)

    keypoints = []

    keypoints = corner_peaks(corner_harris(img1), min_distance=5)

    extractor = BRIEF()

    extractor.extract(img1, keypoints)
    keypoints = keypoints[extractor.mask]
    descriptors = extractor.descriptors
    return  keypoints