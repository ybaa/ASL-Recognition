from skimage.feature import CENSURE as skiCENSURE, BRIEF
from skimage.color import rgb2gray
from attributeExtraction.StandardScaler import __Standard_Scaler__

def CENSURE(image):
    img_orig = rgb2gray(image)

    detector = skiCENSURE()

    detector.detect(img_orig)

    keyPoints = detector.keypoints

    extractor = BRIEF(patch_size=5)

    extractor.extract(img_orig, keyPoints)
    keypoints = keyPoints[extractor.mask]
    descriptors = extractor.descriptors

    return __Standard_Scaler__(descriptors)