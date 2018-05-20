from skimage import transform, novice, color, io
import os


def LoadImages(dirName):
    images = LoadCollectionFromDir(dirName)
    horizontalImages, verticalImages = DevideToHorizontalAndVerticalCollections(images)
    return horizontalImages, verticalImages


def LoadCollectionFromDir(dirName):
    return io.imread_collection(dirName + "/*.jpg")

def DevideToHorizontalAndVerticalCollections(images):
    horizontalImages = []
    verticalImages = []
    for image, file in zip(images, images.files):
        if image.shape[0] > image.shape[1]:
            image = transform.resize(image, (480, 320))
            horizontalImages.append((image, file[11]))
        else:
            image = transform.resize(image, (320, 480))
            verticalImages.append((image, file[11]))

    return horizontalImages, verticalImages




