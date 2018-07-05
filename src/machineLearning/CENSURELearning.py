from src.attributeExtraction.CENSURE import CENSURE
from src.machineLearning.Learner import Extractor


class CENSURE_Extractor(Extractor):

    def __Collection_Extractor__(self, images):
        learnNames = []
        learnKeyPoints = []
        for image in images:
            keyPoints = CENSURE(image[0])
            if len(keyPoints) > 0:
                x = []
                for point in keyPoints:
                    x.append(point)
                if len(x) > 4:
                    learnNames.append(image[1])
                    learnKeyPoints.append(x)
        return learnNames, learnKeyPoints

    def __Individual_Extraction__(self, image):
        return CENSURE(image)
