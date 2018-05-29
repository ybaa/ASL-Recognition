from skimage.color import rgb2gray
from skimage.filters import sobel

def __Sobel__(image):
    newImage = rgb2gray(image)
    edge_sobel = sobel(newImage)
    return edge_sobel
