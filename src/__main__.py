from src.machineLearning.BriefLearning import BriefLerning
from src.machineLearning.ORBLearning import ORBLerning
from src.machineLearning.CENSURELearning import CENSURELerning
from src.machineLearning.Cv2ORBLearning import cv2ORBLerning
from src.imagesOperations.ImagesCollectionLoader import ConcateHorizontalAndVertical

if __name__ == '__main__':

    images = ConcateHorizontalAndVertical("images/smal")

    # BriefLerning(images, 'BriefDataSet')
    # print("Brief learning complete")
    #
    ORBLerning(images, 'ORBDataSet')
    print("ORB learning complete")

    # CENSURELerning(images, 'CENSUREDataSet')
    # print("CENSURE learning complete")

