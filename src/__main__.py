from src.imagesOperations.ImagesCollectionLoader import ConcateHorizontalAndVertical, DevideImagesForTrainingAndTesting
from src.machineLearning.LearningManager import __Testing_learning_parameters__, LearningManager

if __name__ == '__main__':

    learning_Manager = LearningManager(testing=False, c_in=1, gamma_in='auto', decision='ovo')
    gaussianParams = {
        'doGaussian': False,
        'sigma': 1,
        'output': None,
        'mode': 'nearest',
        'cval': 0,
        'multichannel': False,
        'preserve_range': False,
        'truncate': 4.0
    }
    laplaceParams = {
        'doLaplace': False,
        'ksize': 4,
        'mask': None
    }
    images = ConcateHorizontalAndVertical("images/nas", gaussianParams, laplaceParams)
    trainingSet, testingSet = DevideImagesForTrainingAndTesting(images, 0.7)

    # __Testing_learning_parameters__(trainingSet, testingSet)

    learning_Manager.__Learning__(trainingSet)

    learning_Manager.__Save__("ORB_nas_Scalar_laplace")

    learning_Manager.__Tests__(testingSet)
