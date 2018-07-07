from src.attributeExtraction.CENSURE import CENSURE
from src.machineLearning.Learner import Extractor


class CENSUREExtractor(Extractor):

    def collection_extractor(self, images):
        learn_names = []
        learn_key_points = []
        for image in images:
            key_points = CENSURE(image[0])
            if len(key_points) > 0:
                x = []
                for point in key_points:
                    x.append(point)
                if len(x) > 4:
                    learn_names.append(image[1])
                    learn_key_points.append(x)
        return learn_names, learn_key_points

    def individual_extraction(self, image):
        return CENSURE(image)
