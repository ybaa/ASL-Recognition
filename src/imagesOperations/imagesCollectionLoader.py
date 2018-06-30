from skimage import transform, novice, color, io, filters
from medpy.filter.smoothing import anisotropic_diffusion
from imagesOperations.Sobel import __Sobel__


def load_images(dir_name, gaussian_params, laplace_params, anisotropic_params):
    images = load_collection_from_dir(dir_name)
    horizontal_images, vertical_images = devide_to_horizontal_and_vertical_collection(images, gaussian_params, laplace_params,
                                                                                anisotropic_params)
    return horizontal_images, vertical_images


def load_collection_from_dir(dir_name):
    return io.imread_collection(dir_name + "/*.jpg")


def image_conversion(image, gaussian_params, laplace_params, anisotropic_params):
    anisotropic_image = None
    laplace_image = None
    gaussian_image = None
    if anisotropic_params['doAnisotropic']:
        anisotropic_image = anisotropic_diffusion(image,
                                      anisotropic_params['niter'],
                                      anisotropic_params['kappa'],
                                      anisotropic_params['gamma'],
                                      anisotropic_params['voxelspacing'],
                                      anisotropic_params['option'])

    if laplace_params['doLaplace']:
        laplace_image = filters.laplace(image, laplace_params['ksize'], laplace_params['mask'])

    if gaussian_params['doGaussian']:
        gaussian_image = filters.gaussian(image,
                                 gaussian_params['sigma'],
                                 gaussian_params['output'],
                                 gaussian_params['mode'],
                                 gaussian_params['cval'],
                                 gaussian_params['multichannel'],
                                 gaussian_params['preserve_range'],
                                 gaussian_params['truncate'])
    return (anisotropic_image + gaussian_image) / 2


def devide_to_horizontal_and_vertical_collection(images, gaussian_params, laplace_params, anisotropic_params):
    horizontal_images = []
    vertical_images = []
    for image, file in zip(images, images.files):
        if image.shape[0] > image.shape[1]:
            image = transform.resize(image, (480, 320))

            image = image_conversion(image, gaussian_params, laplace_params, anisotropic_params)

            horizontal_images.append((image, file[11]))
        else:
            image = transform.resize(image, (320, 480))

            image = image_conversion(image, gaussian_params, laplace_params, anisotropic_params)

            vertical_images.append((image, file[11]))

    return horizontal_images, vertical_images


def concate_horizontal_and_vertical(src_file, gaussian_params, laplace_params, anisotropic):
    horizontal_images, vertical_images = load_images(src_file, gaussian_params, laplace_params, anisotropic)
    return horizontal_images + vertical_images


# get all images with the same letter and devide it with given percentage
def devide_images_for_training_and_testing(input_set, training_percentage):
    training_set = []
    testing_set = []
    single_letter_set = []

    for i in range(len(input_set) - 1):
        single_letter_set.append(input_set[i])
        if input_set[i][1] != input_set[i + 1][1]:
            training_percent_counter = int(len(single_letter_set) * training_percentage)
            for j in range(len(single_letter_set)):
                if j < training_percent_counter:
                    training_set.append(single_letter_set[j])
                    if j == len(single_letter_set) - 1:
                        single_letter_set.clear()
                else:
                    testing_set.append(single_letter_set[j])
                    if j == len(single_letter_set) - 1:
                        single_letter_set.clear()
    return training_set, testing_set
