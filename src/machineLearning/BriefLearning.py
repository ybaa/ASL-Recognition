from src.attributeExtraction.BRIEFbinaryDescription import BRIEF_skimag
from src.machineLearning.MachineLearning import ReadImageCollection, MinimizeDataSet, GenearteKnowlageBase
import os

def BriefAttributeExtraction(images):
    learnNames = []
    learnKeyPoints = []
    for image, file in zip(images, images.files):
        keyPoints = BRIEF_skimag(image)
        if len(keyPoints) > 0:
            x = []
            for point in keyPoints:
                x.append(point[0])
                x.append(point[1])
            if len(x) > 4:
                learnNames.append(file[11])
                learnKeyPoints.append(x)
    return learnNames, learnKeyPoints

def BriefLerning(srcFile,outputFiel):
    outputFiel = 'BriefDataSet'
    images = ReadImageCollection(srcFile)
    learnNames, learnKeyPoints = BriefAttributeExtraction(images)
    learnKeyPoints = MinimizeDataSet(learnKeyPoints)
    GenearteKnowlageBase(learnKeyPoints,learnNames,outputFiel)