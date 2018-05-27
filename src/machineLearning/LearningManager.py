from src.machineLearning.BriefLearning import Brief_Extractor
from src.machineLearning.CENSURELearning import CENSURE_Extractor
from src.machineLearning.MachineLearning import Knowledge
from src.machineLearning.ORBLearning import ORB_Extractor


class LearningManager:

    def __init__(self):
        self.knowledges = []
        self.knowledges.append(Knowledge(ORB_Extractor(), 'ORB'))
        self.knowledges.append(Knowledge(CENSURE_Extractor(), 'CENSURE'))
        self.knowledges.append(Knowledge(Brief_Extractor(), 'BRIEF'))

    def __Learning__(self, training_set):
        print("LEARNING")
        for knowledge in self.knowledges:
            print(" START " + knowledge.algorithm + " learning")
            knowledge.__Learning__(training_set)
            print(" COMPLETE " + knowledge.algorithm + " learning")
        print("COMPLETE learning")

    def __Tests__(self, testing_set):
        print("TESTING")
        for knowledge in self.knowledges:
            self.__Test__(knowledge, testing_set)
        print("COMPLETE testing")

    def __Test__(self, knowledge, testing_set):
        print(" To " + knowledge.algorithm + ":")
        correctPoint = 0.0
        correctCombine = 0.0
        correctVector = 0.0
        for image in testing_set:
            pointAnswer, combineAnswer, vectorAnswer = knowledge.__Predicting__(image[0], True, True, True)
            if pointAnswer == image[1]:
                correctPoint += 1
            if combineAnswer == image[1]:
                correctCombine += 1
            if vectorAnswer == image[1]:
                correctVector += 1

        testing_Set_len = len(testing_set)
        self.__Test_result__(knowledge.algorithm, correctPoint, testing_Set_len, "Point")
        self.__Test_result__(knowledge.algorithm, correctCombine, testing_Set_len, "Combine")
        self.__Test_result__(knowledge.algorithm, correctVector, testing_Set_len, "Vector")

    def __Test_result__(self, algorithm, corrects, max, name):
        corrects = corrects / max * 100
        print("  " + algorithm + " in " + name + " : ", corrects, "%")
