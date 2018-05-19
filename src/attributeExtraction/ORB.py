import numpy as np
import cv2
from matplotlib import pyplot as plt

def ORB(img):
    # Initiate STAR detector
    orb = cv2.ORB_create()

    # find the keypoints with ORB
    keyPoints = orb.detect(img, None)

    # compute the descriptors with ORB
    keyPoints, des = orb.compute(img, keyPoints)
    return keyPoints

def pictureFromORB(img):
    # draw only keypoints location,not size and orientation
    kp = ORB(img)
    img2 = cv2.drawKeypoints(img, kp, None, color=(0, 255, 0), flags=0)

    plt.imshow(img2), plt.show()