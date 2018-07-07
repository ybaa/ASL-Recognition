from skimage.color import rgb2gray
from skimage.filters import sobel as sob


def sobel(image):
    newImage = rgb2gray(image)
    edge_sobel = sob(newImage)
    return edge_sobel
