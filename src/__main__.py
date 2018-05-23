from src.machineLearning.BriefLearning import __Brief_attribute_extraction__
from src.machineLearning.ORBLearning import __ORB_attribute_extraction__
from src.machineLearning.CENSURELearning import __CENSURE_attribute_extraction__
from src.machineLearning.MachineLearning import Knowledge
from src.imagesOperations.ImagesCollectionLoader import ConcateHorizontalAndVertical, DevideImagesForTrainingAndTesting

if __name__ == '__main__':

    ORB_knowledge = Knowledge()
    BRIEF_knowledge = Knowledge()
    CENSURE_knowledge = Knowledge()
    images = ConcateHorizontalAndVertical("images/big")
    trainingSet, testingSet = DevideImagesForTrainingAndTesting(images, 0.7)

    BRIEF_knowledge.__Learning__(__Brief_attribute_extraction__, images, 'BRIEF')
    print("Brief learning complete")

    ORB_knowledge.__Learning__(__ORB_attribute_extraction__, images, 'ORB')
    print("ORB learning complete")

    CENSURE_knowledge.__Learning__(__CENSURE_attribute_extraction__, images, 'CENSURE')
    print("CENSURE learning complete")

