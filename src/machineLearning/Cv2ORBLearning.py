from src.attributeExtraction.cv2ORB import ORB as cv2ORB
from src.machineLearning.MachineLearning import MinimizeDataSet, GenearteKnowlageBase


def cv2_ORB_attribute_extraction(images):
    learn_names = []
    learn_key_points = []
    for image in images:
        key_points = cv2ORB(image[0])
        x = []
        for point in key_points:
            x.append(point.pt[0])
            x.append(point.pt[1])
        if len(x) > 300:
            learn_names.append(image[1])
            learn_key_points.append(x)
    return learn_names, learn_key_points


def cv2_ORB_learning(images, output_file):
    output_file = 'cv2ORBDataSet'
    learn_names, learn_key_points = cv2_ORB_attribute_extraction(images)
    learn_key_points = MinimizeDataSet(learn_key_points)
    GenearteKnowlageBase(learn_key_points, learn_names, output_file)
