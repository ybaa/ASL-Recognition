from src.attributeExtraction.cv2ORB import ORB as cv2ORB
from src.attributeExtraction.ORB import ORB
from src.machineLearning.MachineLearning import ReadImageCollection, MinimizeDataSet, GenearteKnowlageBase


def cv2ORBAttributeExtraction(images):
    learnNames = []
    learnKeyPoints = []
    for image, file in zip(images, images.files):
        #cv2 ORB version
        keyPoints = cv2ORB(image)
        x = []
        for point in keyPoints:
            x.append(point.pt[0])
            x.append(point.pt[1])
        if len(x) > 300:
            learnNames.append(file[11])
            learnKeyPoints.append(x)
    return learnNames, learnKeyPoints

def cv2ORBLerning(srcFile,outputFiel):
    outputFiel = 'cv2ORBDataSet'
    images = ReadImageCollection(srcFile)
    learnNames, learnKeyPoints = cv2ORBAttributeExtraction(images)
    learnKeyPoints = MinimizeDataSet(learnKeyPoints)
    GenearteKnowlageBase(learnKeyPoints,learnNames,outputFiel)


def ORBAttributeExtraction(images):
    learnNames = []
    learnKeyPoints = []
    for image, file in zip(images, images.files):
        #ORB version
        keyPoints = ORB(image)
        x = []
        for point in keyPoints:
            x.append(point[0])
            x.append(point[1])
        learnNames.append(file[11])
        learnKeyPoints.append(x)
    return learnNames, learnKeyPoints

def ORBLerning(srcFile,outputFiel):
    outputFiel = 'ORBDataSet'
    images = ReadImageCollection(srcFile)
    learnNames, learnKeyPoints = ORBAttributeExtraction(images)
    learnKeyPoints = MinimizeDataSet(learnKeyPoints)
    GenearteKnowlageBase(learnKeyPoints,learnNames,outputFiel)