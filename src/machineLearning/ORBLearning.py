from src.attributeExtraction.ORB import ORB


def __ORB_attribute_extraction__(images):
    learnNames = []
    learnKeyPoints = []
    for image in images:
        keyPoints = ORB(image[0])
        x = []
        for point in keyPoints:
            x.append(point)
        if len(x) > 100:
            learnNames.append(image[1])
            learnKeyPoints.append(x)
    return learnNames, learnKeyPoints
