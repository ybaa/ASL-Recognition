from skimage.color import rgb2gray
from skimage.morphology import skeletonize as skelet


def skeletonize(image):
    # newImage = __rgb2BW_(image)
    newImage = rgb2gray(image)
    skeleton = skelet(newImage)
    return skeleton
