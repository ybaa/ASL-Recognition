from src.machineLearning.BriefLearning import BriefLerning
# from src.machineLearning.ORBLearning import ORBLerning
from src.machineLearning.CENSURELearning import CENSURELerning

if __name__ == '__main__':

    BriefLerning("images/big", 'BriefDataSet')
    print("Brief learning complete")
    #
    # ORBLerning("images/big", 'cv2ORBDataSet')
    # print("ORB learning complete")

    CENSURELerning("images/big", 'cv2ORBDataSet')
    print("CENSURE learning complete")

