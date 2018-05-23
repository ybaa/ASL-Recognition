# from src.machineLearning.BriefLearning import BriefLerning
from src.machineLearning.ORBLearning import __ORB_attribute_extraction__
# from src.machineLearning.CENSURELearning import CENSURELerning
# from src.machineLearning.Cv2ORBLearning import cv2ORBLerning
from src.imagesOperations.ImagesCollectionLoader import ConcateHorizontalAndVertical
from src.machineLearning.MachineLearning import Knowledge

if __name__ == '__main__':

    ORB_knowledge = Knowledge()
    BRIEF_knowledge = Knowledge()
    CENSURE_knowledge = Knowledge()
    images = ConcateHorizontalAndVertical("images/smal")

    # BriefLerning(images, 'BriefDataSet')
    # print("Brief learning complete")
    #
    # ORBLerning(images, 'ORBDataSet')
    ORB_knowledge.__Learning__(__ORB_attribute_extraction__, images, 'ORBDataSet')
    print("ORB learning complete")

    # CENSURELerning(images, 'CENSUREDataSet')
    # print("CENSURE learning complete")

