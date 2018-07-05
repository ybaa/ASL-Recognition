from src.attributeExtraction.cv2ORB import ORB as cv2ORB
from src.machineLearning.MachineLearning import MinimizeDataSet, GenearteKnowlageBase


def cv2ORBAttributeExtraction(images):
    learnNames = []
    learnKeyPoints = []
    for image in images:
        keyPoints = cv2ORB(image[0])
        x = []
        for point in keyPoints:
            x.append(point.pt[0])
            x.append(point.pt[1])
        if len(x) > 300:
            learnNames.append(image[1])
            learnKeyPoints.append(x)
    return learnNames, learnKeyPoints


def cv2ORBLerning(images, outputFile):
    outputFile = 'cv2ORBDataSet'
    learnNames, learnKeyPoints = cv2ORBAttributeExtraction(images)
    learnKeyPoints = MinimizeDataSet(learnKeyPoints)
    GenearteKnowlageBase(learnKeyPoints, learnNames, outputFile)
