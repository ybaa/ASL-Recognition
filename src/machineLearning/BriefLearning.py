from src.attributeExtraction.BRIEFbinaryDescription import BRIEF_skimag
from src.machineLearning.Learner import Extractor


class Brief_Extractor(Extractor):

    def __Collection_Extractor__(self, images):
        learnNames = []
        learnKeyPoints = []
        for image in images:
            keyPoints = BRIEF_skimag(image[0])
            if len(keyPoints) > 0:
                x = []
                for point in keyPoints:
                    x.append(point)
                if len(x) > 4:
                    learnNames.append(image[1])
                    learnKeyPoints.append(x)
        return learnNames, learnKeyPoints

    def __Individual_Extraction__(self, image):
        keyPoints = BRIEF_skimag(image)
        if len(keyPoints) > 0:
            return keyPoints
        else:
            return None