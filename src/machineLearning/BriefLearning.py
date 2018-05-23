from src.attributeExtraction.BRIEFbinaryDescription import BRIEF_skimag

def __Brief_attribute_extraction__(images):
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