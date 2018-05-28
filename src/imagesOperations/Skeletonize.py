from skimage.color import rgb2gray
from skimage.morphology import skeletonize

def __Skeletonize__(image):
    # newImage = __rgb2BW_(image)
    newImage = rgb2gray(image)
    skeleton = skeletonize(newImage)
    return skeleton