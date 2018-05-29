from src.imagesOperations.ImagesCollectionLoader import ConcateHorizontalAndVertical, DevideImagesForTrainingAndTesting
from src.machineLearning.LearningManager import LearningManager

if __name__ == '__main__':

    learning_Manager = LearningManager(testing=False, c_in=2**5, gamma_in='auto', decision='ovo')
    gaussianParams = {
        'doGaussian': True,
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
    anisotropicParams = {
        'doAnisotropic': True,
        'niter': 1,
        'kappa': 50,
        'gamma': 0.1,
        'voxelspacing': None,
        'option': 1
    }
    images = ConcateHorizontalAndVertical("images/nas", gaussianParams, laplaceParams, anisotropicParams)
    trainingSet, testingSet = DevideImagesForTrainingAndTesting(images, 0.7)

    # __Testing_learning_parameters__(trainingSet, testingSet)

    learning_Manager.__Learning__(trainingSet)

    learning_Manager.__Save__("nas_anisotropic_normalize")

    learning_Manager.__Tests__(testingSet)
