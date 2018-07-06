from src.attributeExtraction.BRIEFbinaryDescription import BRIEF_skimag
from src.machineLearning.Learner import Extractor


class BriefExtractor(Extractor):

    def collection_extractor(self, images):
        learn_names = []
        learn_key_points = []
        for image in images:
            key_points = BRIEF_skimag(image[0])
            if len(key_points) > 0:
                x = []
                for point in key_points:
                    x.append(point)
                if len(x) > 4:
                    learn_names.append(image[1])
                    learn_key_points.append(x)
        return learn_names, learn_key_points

    def individual_extraction(self, image):
        key_points = BRIEF_skimag(image)
        if len(key_points) > 0:
            return key_points
        else:
            return None