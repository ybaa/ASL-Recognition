import cv2

from src import ORB
from src.HoG import hog_compute
from src.ORB import pictureFromORB

if __name__ == '__main__':
    image = cv2.imread('images/Machine-Learning-hero.jpg', 0)
    keyPoints = ORB.ORB(image)
    pictureFromORB(image)

    hog = hog_compute(image)
    print("lubie placki")