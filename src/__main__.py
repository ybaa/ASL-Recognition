from src.imagesOperations.ImagesCollectionLoader import ConcateHorizontalAndVertical, DevideImagesForTrainingAndTesting
from src.machineLearning.LearningManager import LearningManager
from src.imagesOperations.ImageRotation import RotateImages

if __name__ == '__main__':
    # RotateImages("images/nas", -4)
    # RotateImages("images/nas", 11)
    # RotateImages("images/nas", -12)

    learning_Manager = LearningManager(testing=False, c_in=2**5, gamma_in='auto', decision='ovo')

    learning_Manager.__Load__("big")

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
    images = ConcateHorizontalAndVertical("images/big", gaussianParams, laplaceParams, anisotropicParams)
    trainingSet, testingSet = DevideImagesForTrainingAndTesting(images, 0.8)

    # __Testing_learning_parameters__(trainingSet, testingSet)

    learning_Manager.__Learning__(trainingSet)

    # learning_Manager.__Save__("small")

    learning_Manager.__Tests__(testingSet)
