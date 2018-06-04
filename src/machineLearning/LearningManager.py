from src.machineLearning.BriefLearning import Brief_Extractor
from src.machineLearning.CENSURELearning import CENSURE_Extractor
from src.machineLearning.MachineLearning import Knowledge
from src.machineLearning.ORBLearning import ORB_Extractor
import pickle


def __Testing_learning_parameters__(trainingSet, testingSet):
    ORB_test_names, ORB_test_key_points = ORB_Extractor().__Collection_Extractor__(testingSet)
    CENSURE_test_names, CENSURE_learn_test_points = CENSURE_Extractor().__Collection_Extractor__(testingSet)

    ORB_test_set = [ORB_test_names, ORB_test_key_points]
    CENSURE_test_set = [CENSURE_test_names, CENSURE_learn_test_points]

    ORB_learn_names, ORB_learn_key_points = ORB_Extractor().__Collection_Extractor__(trainingSet)
    CENSURE_learn_names, CENSURE_learn_key_points = CENSURE_Extractor().__Collection_Extractor__(trainingSet)

    for decision in ['ovo', 'ovr']:
        for c in [-5, 1, 5]:
            gamma = 'auto'
            print("C =", c, " gamma =", gamma, " decision =", decision)
            learningManager = LearningManager(True, 2 ** c, gamma, decision, ORB_learn_names, ORB_learn_key_points,
                                              CENSURE_learn_names, CENSURE_learn_key_points, ORB_test_set,
                                              CENSURE_test_set)

            learningManager.__Learning_test__()

            learningManager.__Tests_test__()


class Dismaches:

    def __init__(self, correct, incorrect):
        self.correct = correct
        self.incorrect = incorrect
        self.multiplication = 1

    def __eq__(self, correct, incorrect):
        return (self.correct == correct) & (self.incorrect == incorrect)

    def increment(self):
        self.multiplication += 1

    def __str__(self):
        string = "correct: " + str(self.correct) + " dismaches with " + str(self.incorrect) + " " + str(self.multiplication) + " times"
        return string


class LearningManager:

    def __init__(self, testing=False, c_in=1, gamma_in='auto', decision='ovr', ORB_learn_names=None,
                 ORB_learn_key_points=None, CENSURE_learn_names=None, CENSURE_learn_key_points=None, ORB_test_set=None,
                 CENSURE_test_set=None):
        if testing:
            self.__Testing_init__(c_in, gamma_in, decision, ORB_learn_names, ORB_learn_key_points, CENSURE_learn_names,
                                  CENSURE_learn_key_points, ORB_test_set, CENSURE_test_set)
        else:
            self.__Regular_init__(c_in, gamma_in, decision)

    def __Regular_init__(self, c_in, gamma_in, decision):
        self.knowledges = []
        # self.knowledges.append(Knowledge(False, CENSURE_Extractor(), 'CENSURE', c_in, gamma_in, decision))
        self.knowledges.append(Knowledge(False, ORB_Extractor(), 'ORB', c_in, gamma_in, decision))
        # self.knowledges.append(Knowledge(False, Brief_Extractor(), 'BRIEF', c_in, gamma_in, decision))

    def __Testing_init__(self, c_in, gamma_in, decision, ORB_learn_names, ORB_learn_key_points, CENSURE_learn_names,
                         CENSURE_learn_key_points, ORB_test_set, CENSURE_test_set):
        self.knowledges = []
        self.knowledges.append(
            Knowledge(True, ORB_Extractor(), 'ORB', c_in, gamma_in, decision, ORB_learn_names, ORB_learn_key_points))
        self.knowledges.append(
            Knowledge(True, CENSURE_Extractor(), 'CENSURE', c_in, gamma_in, decision, CENSURE_learn_names,
                      CENSURE_learn_key_points))
        self.test = []
        self.test.append(ORB_test_set)
        self.test.append(CENSURE_test_set)

    def __Learning__(self, training_set):
        print("LEARNING")
        for knowledge in self.knowledges:
            print(" START " + knowledge.algorithm + " learning")
            knowledge.__Learning__(training_set)
            print(" COMPLETE " + knowledge.algorithm + " learning")
        print("COMPLETE learning")

    def __Learning_test__(self):
        print("LEARNING")
        for knowledge in self.knowledges:
            print(" START " + knowledge.algorithm + " learning")
            knowledge.__Learning_test__()
            print(" COMPLETE " + knowledge.algorithm + " learning")
        print("COMPLETE learning")

    def __Tests__(self, testing_set):
        print("TESTING")
        for knowledge in self.knowledges:
            self.__Test__(knowledge, testing_set)
        print("COMPLETE testing")

    def __Tests_test__(self):
        print("TESTING")
        for knowledge, images in zip(self.knowledges, self.test):
            self.__Test_test__(knowledge, images)
        print("COMPLETE testing")

    def __Test__(self, knowledge, testing_set):
        print(" To " + knowledge.algorithm + ":")
        correctPoint = 0.0
        pointDismaches = []
        correctCombine = 0.0
        combineDismaches = []
        correctVector = 0.0
        vectorDismaches = []
        for image in testing_set:
            try:
                pointAnswer, combineAnswer, vectorAnswer = knowledge.__Predicting__(image[0], True, True, True)
                if pointAnswer == image[1]:
                    correctPoint += 1
                else:
                    pointDismaches = self.Dismaching(pointDismaches, image[1], pointAnswer)
                if combineAnswer == image[1]:
                    correctCombine += 1
                else:
                    combineDismaches = self.Dismaching(combineDismaches, image[1], pointAnswer)
                if vectorAnswer == image[1]:
                    correctVector += 1
                else:
                    vectorDismaches = self.Dismaching(vectorDismaches, image[1], pointAnswer)
            except:
                pass


        testing_Set_len = len(testing_set)
        self.__Test_result__(knowledge.algorithm, correctPoint, testing_Set_len, "Point")
        self.__Test_result__(knowledge.algorithm, correctCombine, testing_Set_len, "Combine")
        self.__Test_result__(knowledge.algorithm, correctVector, testing_Set_len, "Vector")

        print("For Points:")
        for dismache in pointDismaches:
            print(dismache.__str__())
        print("For combine:")
        for dismache in combineDismaches:
            print(dismache.__str__())
        print("For vectors:")
        for dismache in vectorDismaches:
            print(dismache.__str__())

    def Dismaching(self, dimachesColection, correct, incorrect):
        exist = False
        for dismache in dimachesColection:
            if dismache.__eq__(correct, incorrect):
                dismache.increment()
                exist = True
        if not exist:
            dimachesColection.append(Dismaches(correct, incorrect))
        return dimachesColection


    def __Test_test__(self, knowledge, testing_points_set):
        print(" To " + knowledge.algorithm + ":")
        correctPoint = 0.0
        correctCombine = 0.0
        correctVector = 0.0
        for image, name in zip(testing_points_set[1], testing_points_set[0]):
            pointAnswer, combineAnswer, vectorAnswer = knowledge.__Predicting_test__(image, True, True, True)
            if pointAnswer == name:
                correctPoint += 1
            if combineAnswer == name:
                correctCombine += 1
            if vectorAnswer == name:
                correctVector += 1

        testing_Set_len = len(testing_points_set[0])
        self.__Test_result__(knowledge.algorithm, correctPoint, testing_Set_len, "Point")
        self.__Test_result__(knowledge.algorithm, correctCombine, testing_Set_len, "Combine")
        self.__Test_result__(knowledge.algorithm, correctVector, testing_Set_len, "Vector")

    def __Test_result__(self, algorithm, corrects, max, name):
        corrects = corrects / max * 100
        print("  " + algorithm + " in " + name + " : ", corrects, "%")

    def __Save__(self, name):

        with open(name + '.pkl', 'wb') as output:
            pickle.dump(self.__dict__, output, pickle.HIGHEST_PROTOCOL)

    def __Load__(self, name):
        with open(name + '.pkl', 'rb') as input:
            inside = pickle.load(input)
            self.__dict__ = inside
