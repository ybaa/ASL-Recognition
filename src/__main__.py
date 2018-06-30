from imagesOperations.imagesCollectionLoader import concate_horizontal_and_vertical, devide_images_for_training_and_testing
from machineLearning.LearningManager import LearningManager
from imagesOperations.imageRotation import rotate_image


if __name__ == '__main__':
    # RotateImages("images/nas", 15)

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
    images = concate_horizontal_and_vertical("images/tes", gaussianParams, laplaceParams, anisotropicParams)
    training_set, testing_set = devide_images_for_training_and_testing(images, 0.8)

    # __Testing_learning_parameters__(training_set, testing_set)

    learning_Manager.__Learning__(training_set)

    learning_Manager.__Save__("tes_anisotropic_normalize")

    learning_Manager.__Tests__(testing_set)
