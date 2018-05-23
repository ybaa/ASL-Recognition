from sklearn.externals import joblib
from sklearn.svm import SVC
import numpy as np
from src.machineLearning.AlphabetMonitor import AlphabetMonitor


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
    pointClassifier = SVC(kernel='rbf')
    classificationClassifier = SVC(kernel='rbf')
    vectorClassifier = SVC(kernel='rbf')

    newPictures = []
    newNames = []
    for picture, name in zip(learnKeyPoints, learnNames):
        for point in picture:
            newName = (name + '.')[:-1]
            newPictures.append(point)
            newNames.append(newName)

    pointClassifier.fit(newPictures, newNames)

    # learnKeyPoints = MinimizeDataSet(learnKeyPoints)

    names = AlphabetMonitor(learnNames)

    newClassification = []
    for picture in learnKeyPoints:
        answers = []
        for point in picture:
            newPoint = [point]
            answers.append(pointClassifier.predict(newPoint)[0])
        newClassification.append(names.Monitoring(answers))

    classificationClassifier.fit(newClassification, learnNames)

    learnKeyPoints = MinimizeDataSet(learnKeyPoints)

    learnKeyPointsArray = np.array(learnKeyPoints)
    samples, nx, ny = learnKeyPointsArray.shape
    learnKeyPoints = learnKeyPointsArray.reshape((samples, nx*ny))

    vectorClassifier.fit(learnKeyPoints, learnNames)

    with open('point' + outputFile + '.pkl', "wb") as file:
        joblib.dump(pointClassifier, file)

    with open('clasification' + outputFile + '.pkl', "wb") as file:
        joblib.dump(classificationClassifier, file)

    with open('vector' + outputFile + '.pkl', "wb") as file:
        joblib.dump(vectorClassifier, file)
