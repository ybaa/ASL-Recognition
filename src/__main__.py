from src.machineLearning.BriefLearning import __Brief_attribute_extraction__
from src.machineLearning.ORBLearning import __ORB_attribute_extraction__
from src.machineLearning.CENSURELearning import __CENSURE_attribute_extraction__
from src.imagesOperations.ImagesCollectionLoader import ConcateHorizontalAndVertical
from src.machineLearning.MachineLearning import Knowledge

if __name__ == '__main__':

    ORB_knowledge = Knowledge()
    BRIEF_knowledge = Knowledge()
    CENSURE_knowledge = Knowledge()
    images = ConcateHorizontalAndVertical("images/smal")

    # BriefLerning(images, 'BriefDataSet')
    BRIEF_knowledge.__Learning__(__Brief_attribute_extraction__, images, 'BRIEF')
    print("Brief learning complete")

    # ORBLerning(images, 'ORBDataSet')
    ORB_knowledge.__Learning__(__ORB_attribute_extraction__, images, 'ORB')
    print("ORB learning complete")

    # CENSURELerning(images, 'CENSUREDataSet')
    CENSURE_knowledge.__Learning__(__CENSURE_attribute_extraction__, images, 'CENSURE')
    print("CENSURE learning complete")

