from src.attributeExtraction.CENSURE import CENSURE
from src.machineLearning.MachineLearning import MinimizeDataSet, GenearteKnowlageBase


def CENSUREAttributeExtraction(images):
    learnNames = []
    learnKeyPoints = []
    for image in images:
        keyPoints = CENSURE(image[0])
        if len(keyPoints) > 0:
            x = []
            for point in keyPoints:
                x.append(point[0])
                x.append(point[1])
            if len(x) > 4:
                learnNames.append(image[1])
                learnKeyPoints.append(x)
    return learnNames, learnKeyPoints


def CENSURELerning(images, outputFile):
    outputFile = 'CENSUREDataSet'
    learnNames, learnKeyPoints = CENSUREAttributeExtraction(images)
    learnKeyPoints = MinimizeDataSet(learnKeyPoints)
    GenearteKnowlageBase(learnKeyPoints, learnNames, outputFile)
