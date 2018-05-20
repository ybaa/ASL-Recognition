from sklearn.externals import joblib
from sklearn.svm import SVC


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


def GenearteKnowlageBase(learnKeyPoints, learnNames, outputFile):
    clf = SVC()

    clf.fit(learnKeyPoints, learnNames)
    with open(outputFile + '.pkl', "wb") as file:
        joblib.dump(clf, file)
