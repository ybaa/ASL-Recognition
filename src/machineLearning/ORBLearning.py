from src.attributeExtraction.ORB import ORB
from src.machineLearning.Learner import Extractor


class ORBExtractor(Extractor):

    def collection_extractor(self, images):
        learn_names = []
        learn_key_points = []
        for image in images:
            key_points = ORB(image[0])
            x = []
            for point in key_points:
                x.append(point)
            if len(x) > 100:
                learn_names.append(image[1])
                learn_key_points.append(x)
        return learn_names, learn_key_points

    def individual_extraction(self, image):
        return ORB(image)
