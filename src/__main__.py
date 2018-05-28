from src.imagesOperations.ImagesCollectionLoader import ConcateHorizontalAndVertical, DevideImagesForTrainingAndTesting
from src.machineLearning.LearningManager import LearningManager

from src.imagesOperations.Sobel import __Sobel__

if __name__ == '__main__':

    learning_Manager = LearningManager(testing=False, c_in=1, gamma_in='auto', decision='ovo')
    images = ConcateHorizontalAndVertical("images/mid")

    for image in images:
        newImage = __Sobel__(image[0])
        image = list(image)
        image[0] = newImage
        image = tuple(image)

    trainingSet, testingSet = DevideImagesForTrainingAndTesting(images, 0.7)

    # __Testing_learning_parameters__(trainingSet, testingSet)

    learning_Manager.__Learning__(trainingSet)

    learning_Manager.__Tests__(testingSet)
