from src.attributeExtraction.ORB import ORB
from src.machineLearning.Learner import Extractor


class ORB_Extractor(Extractor):

    def __Collection_Extractor__(self, images):
        learnNames = []
        learnKeyPoints = []
        for image in images:
            keyPoints = ORB(image[0])
            x = []
            if keyPoints is not None:
                for point in keyPoints:
                    x.append(point)
                if len(x) > 100:
                    learnNames.append(image[1])
                    learnKeyPoints.append(x)
        return learnNames, learnKeyPoints

    def __Individual_Extraction__(self, image):
        return ORB(image)
