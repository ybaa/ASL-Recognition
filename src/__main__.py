from src.imagesOperations.ImagesCollectionLoader import ConcateHorizontalAndVertical, DevideImagesForTrainingAndTesting
from src.machineLearning.LearningManager import LearningManager

if __name__ == '__main__':

    LearningManager = LearningManager()
    images = ConcateHorizontalAndVertical("images/smal")
    trainingSet, testingSet = DevideImagesForTrainingAndTesting(images, 0.7)

    LearningManager.__Learning__(trainingSet)

    LearningManager.__Tests__(testingSet)

