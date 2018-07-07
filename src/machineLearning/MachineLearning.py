from sklearn.externals import joblib
from sklearn.svm import SVC
import numpy as np
from src.machineLearning.AlphabetMonitor import AlphabetMonitor
from src.machineLearning.Learner import Extractor


def minimize_data_set(learn_key_points):
    min_size = len(learn_key_points[0])
    for points in learn_key_points:
        length = len(points)
        if min_size > length:
            min_size = length

    new_learn_key_points = []
    for points in learn_key_points:
        new_learn_key_points.append(points[:min_size])

    return new_learn_key_points


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
            self.testing_init(manager, name, c_in, gamma_in, decision, learn_names, learn_key_points)
        else:
            self.regular_init(manager, name, c_in, gamma_in, decision)

    def regular_init(self, manager, name, c_in, gamma_in, decision):
        self.AlgorithmManager = manager
        self.pointClassifier = SVC(kernel='rbf', C=c_in, decision_function_shape=decision, gamma=gamma_in)
        self.classificationClassifier = SVC(kernel='rbf', C=c_in, decision_function_shape=decision, gamma=gamma_in)
        self.vectorClassifier = SVC(kernel='rbf', C=c_in, decision_function_shape=decision, gamma=gamma_in)
        self.data_len = 0
        self.algorithm = name

    def testing_init(self, manager, name, c_in, gamma_in, decision, learn_names, learn_key_points):
        self.AlgorithmManager = manager
        self.pointClassifier = SVC(kernel='rbf', C=c_in, decision_function_shape=decision, gamma=gamma_in)
        self.classificationClassifier = SVC(kernel='rbf', C=c_in, decision_function_shape=decision, gamma=gamma_in)
        self.vectorClassifier = SVC(kernel='rbf', C=c_in, decision_function_shape=decision, gamma=gamma_in)
        self.data_len = 0
        self.algorithm = name
        self.learnNames = learn_names
        self.learnKeyPoints = learn_key_points

    def learning_test(self):
        self.names = AlphabetMonitor(self.learnNames)
        self.generate_knowledge_base(self.learnKeyPoints, self.learnNames)

    def learning(self, images):
        learn_names, learn_key_points = self.AlgorithmManager.collection_extractor(images)
        self.names = AlphabetMonitor(learn_names)
        self.generate_knowledge_base(learn_key_points, learn_names)

    def generate_knowledge_base(self, learn_key_points, learn_names):
        # learning point by point
        new_pictures = []
        new_names = []
        for picture, name in zip(learn_key_points, learn_names):
            for point in picture:
                newName = (name + '.')[:-1]
                new_pictures.append(point)
                new_names.append(newName)
        self.pointClassifier.fit(new_pictures, new_names)

        # learning by answers from points
        new_classification = []
        for picture in learn_key_points:
            answers = []
            for point in picture:
                new_point = [point]
                answers.append(self.pointClassifier.predict(new_point)[0])
            new_classification.append(self.names.monitoring(answers))
        self.classificationClassifier.fit(new_classification, learn_names)

        # learning by shortcuts of input
        short_learn_key_points = minimize_data_set(learn_key_points)
        self.data_len = len(short_learn_key_points[0])
        learn_key_points_array = np.array(short_learn_key_points)
        samples, nx, ny = learn_key_points_array.shape
        short_learn_key_points = learn_key_points_array.reshape((samples, nx * ny))
        self.vectorClassifier.fit(short_learn_key_points, learn_names)

        # save data set
        # self.__Save_data_set__()
        # self.__Save__()

    def save_data_set(self, ):
        with open('point' + self.algorithm + '.pkl', "wb") as file:
            joblib.dump(self.pointClassifier, file)

        with open('classification' + self.algorithm + '.pkl', "wb") as file:
            joblib.dump(self.classificationClassifier, file)

        with open('vector' + self.algorithm + '.pkl', "wb") as file:
            joblib.dump(self.vectorClassifier, file)

    def __Save__(self, algorithm_name):
        with open(algorithm_name + '.pkl', "wb") as file:
            joblib.dump(self.names, file)

    def load_data_set(self):
        algorithm_name = self.algorithm
        load_name = 'point' + algorithm_name + '.pkl'
        self.pointClassifier = joblib.load(load_name)

        load_name = algorithm_name + '/classification' + algorithm_name + '.pkl'
        self.classificationClassifier = joblib.load(load_name)

        load_name = algorithm_name + '/vector' + algorithm_name + '.pkl'
        self.vectorClassifier = joblib.load(load_name)

    def load(self, algorithm_name):
        load_name = algorithm_name + '/' + algorithm_name + '.pkl'
        self.names = joblib.load(load_name)
        self.load_data_set(algorithm_name)

    def predicting(self, image, points=False, combine=False, vector=False):
        key__points = self.AlgorithmManager.individual_extraction(image)
        point_answer = ''
        combine_answer = ''
        vector_answer = ''
        if key__points is not None:
            if points or combine:
                answers = []
                for point in key__points:
                    point = [point]
                    answers.append(self.pointClassifier.predict(point))
                counted_answers = self.names.monitoring(answers)
                point_answer = self.names.alphabet[counted_answers.index(max(counted_answers))]
                if combine:
                    combine_answer = self.classificationClassifier.predict([counted_answers])
            if vector and (len(key__points) >= self.data_len):
                key__points = [key__points[:self.data_len]]
                learn_key_points_array = np.array(key__points)
                samples, nx, ny = learn_key_points_array.shape
                key__points = learn_key_points_array.reshape((samples, nx * ny))
                vector_answer = self.vectorClassifier.predict(key__points)

        return point_answer, combine_answer, vector_answer

    def predicting_test(self, key_points, points=False, combine=False, vector=False):
        point_answer = ''
        combine_answer = ''
        vector_answer = ''
        if key_points is not None:
            if points or combine:
                answers = []
                for point in key_points:
                    point = [point]
                    answers.append(self.pointClassifier.predict(point))
                counted_answers = self.names.monitoring(answers)
                point_answer = self.names.alphabet[counted_answers.index(max(counted_answers))]
                if combine:
                    combine_answer = self.classificationClassifier.predict([counted_answers])
            if vector and (len(key_points) >= self.data_len):
                key_points = [key_points[:self.data_len]]
                learn_key_points_array = np.array(key_points)
                samples, nx, ny = learn_key_points_array.shape
                key_points = learn_key_points_array.reshape((samples, nx * ny))
                vector_answer = self.vectorClassifier.predict(key_points)

        return point_answer, combine_answer, vector_answer
