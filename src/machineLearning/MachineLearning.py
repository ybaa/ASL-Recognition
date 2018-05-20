from sklearn.externals import joblib
from sklearn.svm import SVC
from src.imagesOperations.ImagesCollectionLoader import LoadImages

def ReadImageCollection(srcFile):
    horizontalImages, verticalImages = LoadImages(srcFile)
    return horizontalImages + verticalImages

def MinimizeDataSet(learnKeyPoints):
    minSize = len(learnKeyPoints[0])
    for points in learnKeyPoints:
        length = len(points)
        if minSize > length:
            minSize = length

    newLearnKeyPoints = []
    for points in learnKeyPoints:
        newLearnKeyPoints.append(points[:minSize])

    return newLearnKeyPoints

def GenearteKnowlageBase(learnKeyPoints,learnNames,outputFiel):
    clf = SVC()

    clf.fit(learnKeyPoints, learnNames)
    with open(outputFiel + '.pkl', "wb") as file:
        joblib.dump(clf, file)