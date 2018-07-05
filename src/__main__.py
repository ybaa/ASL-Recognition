import sys
sys.path.append('../')

from src.imagesOperations.imagesCollectionLoader import concate_horizontal_and_vertical, devide_images_for_training_and_testing
from src.machineLearning.LearningManager import LearningManager
from src.imagesOperations.imageRotation import rotate_image


if __name__ == '__main__':
    #rotate_image("images/finaltest", 15)

    learning_Manager = LearningManager(testing=False, c_in=2**5, gamma_in='auto', decision='ovo')
    gaussian_params = {
        'doGaussian': True,
        'sigma': 1,
        'output': None,
        'mode': 'nearest',
        'cval': 0,
        'multichannel': False,
        'preserve_range': False,
        'truncate': 4.0
    }
    laplace_params = {
        'doLaplace': False,
        'ksize': 4,
        'mask': None
    }
    anisotropic_params = {
        'doAnisotropic': True,
        'niter': 1,
        'kappa': 50,
        'gamma': 0.1,
        'voxelspacing': None,
        'option': 1
    }
    images = concate_horizontal_and_vertical("images/tes", gaussian_params, laplace_params, anisotropic_params)
    training_set, testing_set = devide_images_for_training_and_testing(images, 0.8)

    # __Testing_learning_parameters__(training_set, testing_set)

    learning_Manager.__Learning__(training_set)

    learning_Manager.__Save__("tes_anisotropic_normalize")

    learning_Manager.__Tests__(testing_set)
