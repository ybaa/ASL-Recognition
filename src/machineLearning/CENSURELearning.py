from src.attributeExtraction.CENSURE import CENSURE
from src.machineLearning.MachineLearning import ReadImageCollection, MinimizeDataSet, GenearteKnowlageBase

def CENSUREAttributeExtraction(images):
    learnNames = []
    learnKeyPoints = []
    for image, file in zip(images, images.files):
        keyPoints = CENSURE(image)
        if len(keyPoints) > 0:
            x = []
            for point in keyPoints:
                x.append(point[0])
                x.append(point[1])
            if len(x) > 4:
                learnNames.append(file[11])
                learnKeyPoints.append(x)
    return learnNames, learnKeyPoints

def CENSURELerning(srcFile,outputFiel):
    outputFiel = 'CENSUREDataSet'
    images = ReadImageCollection(srcFile)
    learnNames, learnKeyPoints = CENSUREAttributeExtraction(images)
    learnKeyPoints = MinimizeDataSet(learnKeyPoints)
    GenearteKnowlageBase(learnKeyPoints,learnNames,outputFiel)