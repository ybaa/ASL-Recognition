from skimage.feature import CENSURE as skiCENSURE, BRIEF
from skimage.color import rgb2gray
from src.attributeExtraction.StandardScaler import standard_scaler

def CENSURE(image):
    img_orig = rgb2gray(image)

    detector = skiCENSURE()

    detector.detect(img_orig)

    keyPoints = detector.keypoints

    extractor = BRIEF(patch_size=5)

    extractor.extract(img_orig, keyPoints)
    keypoints = keyPoints[extractor.mask]
    descriptors = extractor.descriptors

    return standard_scaler(descriptors)