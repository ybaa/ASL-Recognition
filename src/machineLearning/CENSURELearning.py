from src.attributeExtraction.CENSURE import CENSURE


def __CENSURE_attribute_extraction__(images):
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
