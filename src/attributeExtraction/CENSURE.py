from skimage.feature import CENSURE as skiCENSURE
from skimage.color import rgb2gray

def CENSURE(image):
    img_orig = rgb2gray(image)

    detector = skiCENSURE()

    detector.detect(img_orig)

    keyPoints = detector.keypoints
    return keyPoints