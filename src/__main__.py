from src.machineLearning.BriefLearning import __Brief_attribute_extraction__
from src.machineLearning.ORBLearning import ORB_Extractor
from src.machineLearning.CENSURELearning import __CENSURE_attribute_extraction__
from src.machineLearning.MachineLearning import Knowledge
from src.imagesOperations.ImagesCollectionLoader import ConcateHorizontalAndVertical, DevideImagesForTrainingAndTesting

if __name__ == '__main__':

    ORB_knowledge = Knowledge(ORB_Extractor())
    # ORB_knowledge.__Load__('ORB')
    # BRIEF_knowledge = Knowledge()
    # CENSURE_knowledge = Knowledge()
    images = ConcateHorizontalAndVertical("images/midium")
    trainingSet, testingSet = DevideImagesForTrainingAndTesting(images, 0.7)

    # BRIEF_knowledge.__Learning__(__Brief_attribute_extraction__, images, 'BRIEF')
    # print("Brief learning complete")

    ORB_knowledge.__Learning__(trainingSet, 'ORB')
    print("ORB learning complete")

    # CENSURE_knowledge.__Learning__(__CENSURE_attribute_extraction__, images, 'CENSURE')
    # print("CENSURE learning complete")

    print("TESTING")
    print("To ORB:")
    correctPoint = 0.0
    correctCombine = 0.0
    correctVector = 0.0
    for image in testingSet:
        pointAnswer, combineAnswer, vectorAnswer = ORB_knowledge.__Predicting__(image[0], True, True, True)
        if pointAnswer == image[1]:
            correctPoint += 1
        if combineAnswer == image[1]:
            correctCombine += 1
        if vectorAnswer == image[1]:
            correctVector += 1

    correctPoint = correctPoint/len(testingSet)
    correctCombine = correctCombine/len(testingSet)
    correctVector = correctVector/len(testingSet)
    print("CORRECT in Point : ", correctPoint, "%")
    print("CORRECT in Combine : ", correctCombine, "%")
    print("CORRECT in Vector : ", correctVector, "%")
