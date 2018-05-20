from skimage import data
from skimage.feature import CENSURE as skiCENSURE
from skimage.color import rgb2gray

import matplotlib.pyplot as plt

def CENSURE(image):
    img_orig = rgb2gray(data.astronaut())

    detector = skiCENSURE()

    detector.detect(img_orig)

    keyPoints = detector.keypoints
    return keyPoints