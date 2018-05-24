from copy import copy

from src.machineLearning.BriefLearning import Brief_Extractor
from src.machineLearning.ORBLearning import ORB_Extractor
from src.machineLearning.CENSURELearning import CENSURE_Extractor
from src.machineLearning.MachineLearning import Knowledge
from src.imagesOperations.ImagesCollectionLoader import ConcateHorizontalAndVertical, DevideImagesForTrainingAndTesting


def tests(algorithm, knowledge):
    print("To " + algorithm + ":")
    correctPoint = 0.0
    correctCombine = 0.0
    correctVector = 0.0
    for image in testingSet:
        pointAnswer, combineAnswer, vectorAnswer = knowledge.__Predicting__(image[0], True, True, True)
        if pointAnswer == image[1]:
            correctPoint += 1
        if combineAnswer == image[1]:
            correctCombine += 1
        if vectorAnswer == image[1]:
            correctVector += 1

    correctPoint = correctPoint/len(testingSet)
    correctCombine = correctCombine/len(testingSet)
    correctVector = correctVector/len(testingSet)
    print(algorithm + " in Point : ", correctPoint, "%")
    print(algorithm + " in Combine : ", correctCombine, "%")
    print(algorithm + " in Vector : ", correctVector, "%")

if __name__ == '__main__':

    ORB_knowledge = Knowledge(ORB_Extractor())
    CENSURE_knowledge = Knowledge(CENSURE_Extractor())
    BRIEF_knowledge = Knowledge(Brief_Extractor())
    # ORB_knowledge.__Load__('ORB')
    images = ConcateHorizontalAndVertical("images/big")
    trainingSet, testingSet = DevideImagesForTrainingAndTesting(images, 0.7)

    BRIEF_knowledge.__Learning__(images, 'BRIEF')
    print("Brief learning complete")

    ORB_knowledge.__Learning__(trainingSet, 'ORB')
    print("ORB learning complete")

    CENSURE_knowledge.__Learning__(images, 'CENSURE')
    print("CENSURE learning complete")

    print("TESTING")
    tests('ORB', ORB_knowledge)
    tests('CENSURE', CENSURE_knowledge)
    tests('BRIEF', BRIEF_knowledge)


