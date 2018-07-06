import numpy as np


def rgb_2_bw(image):
    newImage = []
    for row in image:
        newRow = []
        for point in row:
            value = sum(point)/3
            if value >= 0.5:
                value = 1
            else:
                value = 0
            newRow.append(value)
        newImage.append(np.array(newRow))
    newImage = np.array(newImage)
    return newImage