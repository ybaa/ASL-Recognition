from sklearn.externals import joblib
from sklearn.svm import SVC
import numpy as np
from src.machineLearning.AlphabetMonitor import AlphabetMonitor
from src.machineLearning.Learner import Extractor


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
    pointClassifier = None
    classificationClassifier = None
    vectorClassifier = None
    data_len = 0
    AlgorithmManager = None
    names = None

    def __init__(self, testing=False, manager=None, name=None, c_in=None, gamma_in=None, decision=None,
                 learn_names=None, learn_key_points=None):
        if testing:
            self.__Testing_init__(manager, name, c_in, gamma_in, decision, learn_names, learn_key_points)
        else:
            self.__Regular_init__(manager, name, c_in, gamma_in, decision)

    def __Regular_init__(self, manager, name, c_in, gamma_in, decision):
        self.AlgorithmManager = manager
        self.pointClassifier = SVC(kernel='rbf', C=c_in, decision_function_shape=decision, gamma=gamma_in)
        self.classificationClassifier = SVC(kernel='rbf', C=c_in, decision_function_shape=decision, gamma=gamma_in)
        self.vectorClassifier = SVC(kernel='rbf', C=c_in, decision_function_shape=decision, gamma=gamma_in)
        self.data_len = 0
        self.algorithm = name

    def __Testing_init__(self, manager, name, c_in, gamma_in, decision, learn_names, learn_key_points):
        self.AlgorithmManager = manager
        self.pointClassifier = SVC(kernel='rbf', C=c_in, decision_function_shape=decision, gamma=gamma_in)
        self.classificationClassifier = SVC(kernel='rbf', C=c_in, decision_function_shape=decision, gamma=gamma_in)
        self.vectorClassifier = SVC(kernel='rbf', C=c_in, decision_function_shape=decision, gamma=gamma_in)
        self.data_len = 0
        self.algorithm = name
        self.learnNames = learn_names
        self.learnKeyPoints = learn_key_points

    def __Learning_test__(self):
        self.names = AlphabetMonitor(self.learnNames)
        self.__Generate_Knowledge_Base__(self.learnKeyPoints, self.learnNames)

    def __Learning__(self, images):
        learnNames, learnKeyPoints = self.AlgorithmManager.__Collection_Extractor__(images)
        self.names = AlphabetMonitor(learnNames)
        self.__Generate_Knowledge_Base__(learnKeyPoints, learnNames)

    def __Generate_Knowledge_Base__(self, learn_key_points, learn_names):
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
        newClassification = []
        for picture in learn_key_points:
            answers = []
            for point in picture:
                newPoint = [point]
                answers.append(self.pointClassifier.predict(newPoint)[0])
            newClassification.append(self.names.__Monitoring__(answers))
        self.classificationClassifier.fit(newClassification, learn_names)

        # learning by shortcuts of input
        shortLearnKeyPoints = __Minimize_Data_Set__(learn_key_points)
        self.data_len = len(shortLearnKeyPoints[0])
        learnKeyPointsArray = np.array(shortLearnKeyPoints)
        samples, nx, ny = learnKeyPointsArray.shape
        shortLearnKeyPoints = learnKeyPointsArray.reshape((samples, nx * ny))
        self.vectorClassifier.fit(shortLearnKeyPoints, learn_names)

        # save data set
        self.__Save_data_set__()
        # self.__Save__()

    def __Save_data_set__(self, ):
        with open('point' + self.algorithm + '.pkl', "wb") as file:
            joblib.dump(self.pointClassifier, file)

        with open('classification' + self.algorithm + '.pkl', "wb") as file:
            joblib.dump(self.classificationClassifier, file)

        with open('vector' + self.algorithm + '.pkl', "wb") as file:
            joblib.dump(self.vectorClassifier, file)

    def __Save__(self, algorithm_name):
        with open(algorithm_name + '.pkl', "wb") as file:
            joblib.dump(self.names, file)

    def __Load_data_set__(self):
        algorithm_name = self.algorithm
        load_name = 'point' + algorithm_name + '.pkl'
        self.pointClassifier = joblib.load(load_name)

        load_name = algorithm_name + '/classification' + algorithm_name + '.pkl'
        self.classificationClassifier = joblib.load(load_name)

        load_name = algorithm_name + '/vector' + algorithm_name + '.pkl'
        self.vectorClassifier = joblib.load(load_name)

    def __Load__(self, algorithm_name):
        load_name = algorithm_name + '/' + algorithm_name + '.pkl'
        self.names = joblib.load(load_name)
        self.__Load_data_set__(algorithm_name)

    def __Predicting__(self, image, points=False, combine=False, vector=False):
        key_Points = self.AlgorithmManager.__Individual_Extraction__(image)
        pointAnswer = ''
        combineAnswer = ''
        vectorAnswer = ''
        if key_Points is not None:
            if points or combine:
                answers = []
                for point in key_Points:
                    point = [point]
                    answers.append(self.pointClassifier.predict(point))
                countedAnswers = self.names.__Monitoring__(answers)
                pointAnswer = self.names.alphabet[countedAnswers.index(max(countedAnswers))]
                if combine:
                    combineAnswer = self.classificationClassifier.predict([countedAnswers])
            if vector and (len(key_Points) >= self.data_len):
                key_Points = [key_Points[:self.data_len]]
                learnKeyPointsArray = np.array(key_Points)
                samples, nx, ny = learnKeyPointsArray.shape
                key_Points = learnKeyPointsArray.reshape((samples, nx * ny))
                vectorAnswer = self.vectorClassifier.predict(key_Points)

        return pointAnswer, combineAnswer, vectorAnswer

    def __Predicting_test__(self, key_Points, points=False, combine=False, vector=False):
        pointAnswer = ''
        combineAnswer = ''
        vectorAnswer = ''
        if key_Points is not None:
            if points or combine:
                answers = []
                for point in key_Points:
                    point = [point]
                    answers.append(self.pointClassifier.predict(point))
                countedAnswers = self.names.__Monitoring__(answers)
                pointAnswer = self.names.alphabet[countedAnswers.index(max(countedAnswers))]
                if combine:
                    combineAnswer = self.classificationClassifier.predict([countedAnswers])
            if vector and (len(key_Points) >= self.data_len):
                key_Points = [key_Points[:self.data_len]]
                learnKeyPointsArray = np.array(key_Points)
                samples, nx, ny = learnKeyPointsArray.shape
                key_Points = learnKeyPointsArray.reshape((samples, nx * ny))
                vectorAnswer = self.vectorClassifier.predict(key_Points)

        return pointAnswer, combineAnswer, vectorAnswer
