from src.attributeExtraction.ORB import ORB
from src.machineLearning.MachineLearning import MinimizeDataSet, GenearteKnowlageBase

def ORBAttributeExtraction(images):
    learnNames = []
    learnKeyPoints = []
    for image in images:
        keyPoints = ORB(image[0])
        x = []
        for point in keyPoints:
            x.append(point)
            # x.append(point[0])
            # x.append(point[1])
        if len(x)>100:
            learnNames.append(image[1])
            learnKeyPoints.append(x)
    return learnNames, learnKeyPoints


def ORBLerning(images, outputFile):
    outputFile = 'ORBDataSet'
    learnNames, learnKeyPoints = ORBAttributeExtraction(images)
    # learnKeyPoints = MinimizeDataSet(learnKeyPoints)
    GenearteKnowlageBase(learnKeyPoints, learnNames, outputFile)
