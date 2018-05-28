from skimage.feature import ORB as skiORB
from skimage.color import rgb2gray
from src.attributeExtraction.StandardScaler import __Standard_Scaler__


def ORB(img):
    img1 = rgb2gray(img)

    descriptor_extractor = skiORB(n_keypoints=200)

    descriptor_extractor.detect_and_extract(img1)
    keypoints = descriptor_extractor.keypoints
    descriptors = descriptor_extractor.descriptors

    return __Standard_Scaler__(descriptors)