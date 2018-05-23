from sklearn.externals import joblib
from sklearn.svm import SVC
import numpy as np
from src.machineLearning.AlphabetMonitor import AlphabetMonitor


def __Minimize_Data_Set__(learn_key_points):
    minSize = len(learn_key_points[0])
    for points in learn_key_points:
        length = len(points)
        if minSize > length:
            minSize = length

    newLearnKeyPoints = []
    for points in learn_key_points:
        newLearnKeyPoints.append(points[:minSize])

    return newLearnKeyPoints


class Knowledge:
    pointClassifier = SVC(kernel='rbf')
    classificationClassifier = SVC(kernel='rbf')
    vectorClassifier = SVC(kernel='rbf')
    data_len = 0

    def __Learning__(self, extractor, images, output_file):
        self.Algorithm = extractor
        learnNames, learnKeyPoints = self.Algorithm(images)
        self.__Generate_Knowledge_Base__(learnKeyPoints, learnNames, output_file)

    def __Generate_Knowledge_Base__(self, learn_key_points, learn_names, output_file):
        # learning point by point
        newPictures = []
        newNames = []
        for picture, name in zip(learn_key_points, learn_names):
            for point in picture:
                newName = (name + '.')[:-1]
                newPictures.append(point)
                newNames.append(newName)
        self.pointClassifier.fit(newPictures, newNames)

        # learning by answers from points
        names = AlphabetMonitor(learn_names)
        newClassification = []
        for picture in learn_key_points:
            answers = []
            for point in picture:
                newPoint = [point]
                answers.append(self.pointClassifier.predict(newPoint)[0])
            newClassification.append(names.__Monitoring__(answers))
        self.classificationClassifier.fit(newClassification, learn_names)

        # learning by shortcuts of input
        shortLearnKeyPoints = __Minimize_Data_Set__(learn_key_points)
        learnKeyPointsArray = np.array(shortLearnKeyPoints)
        samples, nx, ny = learnKeyPointsArray.shape
        shortLearnKeyPoints = learnKeyPointsArray.reshape((samples, nx * ny))
        self.vectorClassifier.fit(shortLearnKeyPoints, learn_names)
        # save data set
        self.__Save_data_set__(output_file)

    def __Save_data_set__(self, algorithm_name):
        with open('point' + algorithm_name + '.pkl', "wb") as file:
            joblib.dump(self.pointClassifier, file)

        with open('classification' + algorithm_name + '.pkl', "wb") as file:
            joblib.dump(self.classificationClassifier, file)

        with open('vector' + algorithm_name + '.pkl', "wb") as file:
            joblib.dump(self.vectorClassifier, file)

    def __Load_data_set__(self, algorithm_name):
        load_name = 'point' + algorithm_name + '.pkl'
        self.pointClassifier = joblib.load(load_name)

        load_name = 'classification' + algorithm_name + '.pkl'
        self.classificationClassifier = joblib.load(load_name)

        load_name = 'vector' + algorithm_name + '.pkl'
        self.vectorClassifier = joblib.load(load_name)
