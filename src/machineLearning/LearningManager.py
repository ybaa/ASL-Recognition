from src.machineLearning.BriefLearning import BriefExtractor
from src.machineLearning.CENSURELearning import CENSUREExtractor
from src.machineLearning.MachineLearning import Knowledge
from src.machineLearning.ORBLearning import ORBExtractor
import pickle


def testing_learning_parameters(trainingSet, testingSet):
    ORB_test_names, ORB_test_key_points = ORBExtractor().collection_extractor(testingSet)
    CENSURE_test_names, CENSURE_learn_test_points = CENSUREExtractor().collection_extractor(testingSet)

    ORB_test_set = [ORB_test_names, ORB_test_key_points]
    CENSURE_test_set = [CENSURE_test_names, CENSURE_learn_test_points]

    ORB_learn_names, ORB_learn_key_points = ORBExtractor().collection_extractor(trainingSet)
    CENSURE_learn_names, CENSURE_learn_key_points = CENSUREExtractor().collection_extractor(trainingSet)

    for decision in ['ovo', 'ovr']:
        for c in [-5, 1, 5]:
            gamma = 'auto'
            print("C =", c, " gamma =", gamma, " decision =", decision)
            learningManager = LearningManager(True, 2 ** c, gamma, decision, ORB_learn_names, ORB_learn_key_points,
                                              CENSURE_learn_names, CENSURE_learn_key_points, ORB_test_set,
                                              CENSURE_test_set)

            learningManager.learning_test()

            learningManager.tests_test()


class LearningManager:

    def __init__(self, testing=False, c_in=1, gamma_in='auto', decision='ovr', ORB_learn_names=None,
                 ORB_learn_key_points=None, CENSURE_learn_names=None, CENSURE_learn_key_points=None, ORB_test_set=None,
                 CENSURE_test_set=None):
        if testing:
            self.testing_init(c_in, gamma_in, decision, ORB_learn_names, ORB_learn_key_points, CENSURE_learn_names,
                              CENSURE_learn_key_points, ORB_test_set, CENSURE_test_set)
        else:
            self.regular_init(c_in, gamma_in, decision)

    def regular_init(self, c_in, gamma_in, decision):
        self.knowledges = []
        # self.knowledges.append(Knowledge(False, CENSURE_Extractor(), 'CENSURE', c_in, gamma_in, decision))
        self.knowledges.append(Knowledge(False, ORBExtractor(), 'ORB', c_in, gamma_in, decision))
        # self.knowledges.append(Knowledge(False, Brief_Extractor(), 'BRIEF', c_in, gamma_in, decision))

    def testing_init(self, c_in, gamma_in, decision, ORB_learn_names, ORB_learn_key_points, CENSURE_learn_names,
                     CENSURE_learn_key_points, ORB_test_set, CENSURE_test_set):
        self.knowledges = []
        self.knowledges.append(
            Knowledge(True, ORBExtractor(), 'ORB', c_in, gamma_in, decision, ORB_learn_names, ORB_learn_key_points))
        self.knowledges.append(
            Knowledge(True, CENSUREExtractor(), 'CENSURE', c_in, gamma_in, decision, CENSURE_learn_names,
                      CENSURE_learn_key_points))
        self.test = []
        self.test.append(ORB_test_set)
        self.test.append(CENSURE_test_set)

    def learning(self, training_set):
        print("LEARNING")
        for knowledge in self.knowledges:
            print(" START " + knowledge.algorithm + " learning")
            knowledge.learning(training_set)
            print(" COMPLETE " + knowledge.algorithm + " learning")
        print("COMPLETE learning")

    def learning_test(self):
        print("LEARNING")
        for knowledge in self.knowledges:
            print(" START " + knowledge.algorithm + " learning")
            knowledge.learning_test()
            print(" COMPLETE " + knowledge.algorithm + " learning")
        print("COMPLETE learning")

    def tests(self, testing_set):
        print("TESTING")
        for knowledge in self.knowledges:
            self.test(knowledge, testing_set)
        print("COMPLETE testing")

    def tests_test(self):
        print("TESTING")
        for knowledge, images in zip(self.knowledges, self.test):
            self.test_test(knowledge, images)
        print("COMPLETE testing")

    def test(self, knowledge, testing_set):
        print(" To " + knowledge.algorithm + ":")
        correct_point = 0.0
        correct_combine = 0.0
        correct_vector = 0.0
        for image in testing_set:
            point_answer, combine_answer, vector_answer = knowledge.predicting(image[0], True, True, True)
            if point_answer == image[1]:
                correct_point += 1
            if combine_answer == image[1]:
                correct_combine += 1
            if vector_answer == image[1]:
                correct_vector += 1

        testing__set_len = len(testing_set)
        self.test_result(knowledge.algorithm, correct_point, testing__set_len, "Point")
        self.test_result(knowledge.algorithm, correct_combine, testing__set_len, "Combine")
        self.test_result(knowledge.algorithm, correct_vector, testing__set_len, "Vector")

    def test_test(self, knowledge, testing_points_set):
        print(" To " + knowledge.algorithm + ":")
        correct_point = 0.0
        correct_combine = 0.0
        correct_vector = 0.0
        for image, name in zip(testing_points_set[1], testing_points_set[0]):
            point_answer, combine_answer, vector_answer = knowledge.predicting_test(image, True, True, True)
            if point_answer == name:
                correct_point += 1
            if combine_answer == name:
                correct_combine += 1
            if vector_answer == name:
                correct_vector += 1

        testing__set_len = len(testing_points_set[0])
        self.test_result(knowledge.algorithm, correct_point, testing__set_len, "Point")
        self.test_result(knowledge.algorithm, correct_combine, testing__set_len, "Combine")
        self.test_result(knowledge.algorithm, correct_vector, testing__set_len, "Vector")

    def test_result(self, algorithm, corrects, max, name):
        corrects = corrects / max * 100
        print("  " + algorithm + " in " + name + " : ", corrects, "%")

    def save(self, name):

        with open(name + '.pkl', 'wb') as output:
            pickle.dump(self.__dict__, output, pickle.HIGHEST_PROTOCOL)

    def load(self, name):
        with open(name + '.pkl', 'rb') as input:
            inside = pickle.load(input)
            self.__dict__ = inside
