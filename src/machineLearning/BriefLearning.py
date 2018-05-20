from src.attributeExtraction.BRIEFbinaryDescription import BRIEF_skimag
from src.machineLearning.MachineLearning import MinimizeDataSet, GenearteKnowlageBase

def BriefAttributeExtraction(images):
    learnNames = []
    learnKeyPoints = []
    for image in images:
        keyPoints = BRIEF_skimag(image[0])
        if len(keyPoints) > 0:
            x = []
            for point in keyPoints:
                x.append(point[0])
                x.append(point[1])
            if len(x) > 4:
                learnNames.append(image[1])
                learnKeyPoints.append(x)
    return learnNames, learnKeyPoints

def BriefLerning(images, outputFile):
    outputFile = 'BriefDataSet'
    learnNames, learnKeyPoints = BriefAttributeExtraction(images)
    learnKeyPoints = MinimizeDataSet(learnKeyPoints)
    GenearteKnowlageBase(learnKeyPoints, learnNames, outputFile)