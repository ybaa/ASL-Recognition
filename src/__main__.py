from src.machineLearning.BriefLearning import BriefLerning
# from src.machineLearning.ORBLearning import cv2ORBLerning

if __name__ == '__main__':

    BriefLerning("images/big", 'BriefDataSet')
    print("Brief learning complete")

    # cv2ORBLerning("images/big", 'cv2ORBDataSet')
    # print("vc2 ORB learning complete")
