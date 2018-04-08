import datetime

import cv2
import imutils as imutils
from matplotlib import pyplot as plt


# load the image and resize it
import numpy as np
from imutils import feature

image = cv2.imread("images/Machine-Learning-hero.jpg")
image = imutils.resize(image, width=min(400, image.shape[1]))

def hog_compute(ims):
    samples=[]
    winSize = (64,64)
    blockSize = (16,16)
    blockStride = (8,8)
    cellSize = (8,8)
    nbins = 9
    derivAperture = 1
    winSigma = 4.
    histogramNormType = 0
    L2HysThreshold = 2.0000000000000001e-01
    gammaCorrection = 0
    nlevels = 64
    hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,
                            histogramNormType,L2HysThreshold,gammaCorrection,nlevels)
    #compute(img[, winStride[, padding[, locations]]]) -> descriptors
    winStride = (8,8)
    padding = (8,8)
    locations = ((10,20),(30,30),(50,50),(70,70),(90,90),(110,110),(130,130),(150,150),(170,170),(190,190))
    ###########################
    # version for many images #
    ###########################
    # for im in ims:
    #     hist = hog.compute(im,winStride,padding,locations)
    #     samples.append(hist)
    # return np.float32(samples)
    ##########################
    # version for one images #
    ##########################
    hist = hog.compute(image,winStride,padding,locations)
    return np.float32(hist)

def histogramVisualization(hist):
    plt.hist(hist)
    plt.show();