from src.imagesOperations.ImagesCollectionLoader import ConcateHorizontalAndVertical, DevideImagesForTrainingAndTesting
from src.machineLearning.LearningManager import __Testing_learning_parameters__, LearningManager

if __name__ == '__main__':

    learning_Manager = LearningManager(testing=False, c_in=1, gamma_in='auto', decision='ovo')
    images = ConcateHorizontalAndVertical("images/sma")
    trainingSet, testingSet = DevideImagesForTrainingAndTesting(images, 0.7)

    # __Testing_learning_parameters__(trainingSet, testingSet)

    learning_Manager.__Learning__(trainingSet)

    learning_Manager.__Tests__(testingSet)
