from skimage import transform, novice, color, io, filters
import os


def LoadImages(dirName, doGaussianlur, gaussianParams):
    images = LoadCollectionFromDir(dirName)
    horizontalImages, verticalImages = DevideToHorizontalAndVerticalCollections(images, doGaussianlur, gaussianParams)
    return horizontalImages, verticalImages


def LoadCollectionFromDir(dirName):
    return io.imread_collection(dirName + "/*.jpg")

def DevideToHorizontalAndVerticalCollections(images, doGaussianBlur, gaussianParams):
    horizontalImages = []
    verticalImages = []
    for image, file in zip(images, images.files):
        if image.shape[0] > image.shape[1]:
            image = transform.resize(image, (480, 320))
            if(doGaussianBlur):
                filters.gaussian(image,
                                 gaussianParams['sigma'],
                                 gaussianParams['output'],
                                 gaussianParams['mode'],
                                 gaussianParams['cval'],
                                 gaussianParams['multichannel'],
                                 gaussianParams['preserve_range'],
                                 gaussianParams['truncate'], )
            horizontalImages.append((image, file[11]))
        else:
            image = transform.resize(image, (320, 480))
            if (doGaussianBlur):
                filters.gaussian(image,
                                 gaussianParams['sigma'],
                                 gaussianParams['output'],
                                 gaussianParams['mode'],
                                 gaussianParams['cval'],
                                 gaussianParams['multichannel'],
                                 gaussianParams['preserve_range'],
                                 gaussianParams['truncate'], )
            verticalImages.append((image, file[11]))

    return horizontalImages, verticalImages


def ConcateHorizontalAndVertical(srcFile, doGaussianBlur, gaussianParams):
    horizontalImages, verticalImages = LoadImages(srcFile, doGaussianBlur, gaussianParams)
    return horizontalImages + verticalImages


#get all images with the same letter and devide it with given percentage
def DevideImagesForTrainingAndTesting(inputSet, trainingPercent):
    trainingSet = []
    testingSet = []
    singleLetterSet = []

    for i in range(len(inputSet)-1):
        singleLetterSet.append(inputSet[i])
        if inputSet[i][1] != inputSet[i+1][1]:
            trainingPercentCounter = int(len(singleLetterSet) * trainingPercent)
            for j in range(len(singleLetterSet)):
                if j < trainingPercentCounter:
                    trainingSet.append(singleLetterSet[j])
                    if j == len(singleLetterSet) - 1:
                        singleLetterSet.clear()
                else:
                    testingSet.append(singleLetterSet[j])
                    if j == len(singleLetterSet)-1:
                        singleLetterSet.clear()
    return trainingSet, testingSet


