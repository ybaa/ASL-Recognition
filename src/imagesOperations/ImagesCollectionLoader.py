from skimage import transform, novice, color, io, filters
from medpy.filter.smoothing import anisotropic_diffusion
from src.imagesOperations.Sobel import __Sobel__


def LoadImages(dirName, gaussianParams, laplaceParams, anisotropicParams):
    images = LoadCollectionFromDir(dirName)
    horizontalImages, verticalImages = DevideToHorizontalAndVerticalCollections(images, gaussianParams, laplaceParams,
                                                                                anisotropicParams)
    return horizontalImages, verticalImages


def LoadCollectionFromDir(dirName):
    return io.imread_collection(dirName + "/*.jpg")


def ImageConvwersions(image, gaussianParams, laplaceParams, anisotropicParams):
    anisotropicImage = None
    laplaceImage = None
    goussianImage = None
    if anisotropicParams['doAnisotropic']:
        anisotropicImage = anisotropic_diffusion(image,
                                      anisotropicParams['niter'],
                                      anisotropicParams['kappa'],
                                      anisotropicParams['gamma'],
                                      anisotropicParams['voxelspacing'],
                                      anisotropicParams['option'])

    if laplaceParams['doLaplace']:
        laplaceImage = filters.laplace(image, laplaceParams['ksize'], laplaceParams['mask'])

    if gaussianParams['doGaussian']:
        goussianImage = filters.gaussian(image,
                                 gaussianParams['sigma'],
                                 gaussianParams['output'],
                                 gaussianParams['mode'],
                                 gaussianParams['cval'],
                                 gaussianParams['multichannel'],
                                 gaussianParams['preserve_range'],
                                 gaussianParams['truncate'])
    return (anisotropicImage + goussianImage) / 2


def DevideToHorizontalAndVerticalCollections(images, gaussianParams, laplaceParams, anisotropicParams):
    horizontalImages = []
    verticalImages = []
    for image, file in zip(images, images.files):
        if image.shape[0] > image.shape[1]:
            image = transform.resize(image, (480, 320))

            image = ImageConvwersions(image, gaussianParams, laplaceParams, anisotropicParams)

            horizontalImages.append((image, file[11]))
        else:
            image = transform.resize(image, (320, 480))

            image = ImageConvwersions(image, gaussianParams, laplaceParams, anisotropicParams)

            verticalImages.append((image, file[11]))

    return horizontalImages, verticalImages


def ConcateHorizontalAndVertical(srcFile, gaussianParams, laplaceParams, anisotropic):
    horizontalImages, verticalImages = LoadImages(srcFile, gaussianParams, laplaceParams, anisotropic)
    return horizontalImages + verticalImages


# get all images with the same letter and devide it with given percentage
def DevideImagesForTrainingAndTesting(inputSet, trainingPercent):
    trainingSet = []
    testingSet = []
    singleLetterSet = []

    for i in range(len(inputSet) - 1):
        singleLetterSet.append(inputSet[i])
        if inputSet[i][1] != inputSet[i + 1][1]:
            trainingPercentCounter = int(len(singleLetterSet) * trainingPercent)
            for j in range(len(singleLetterSet)):
                if j < trainingPercentCounter:
                    trainingSet.append(singleLetterSet[j])
                    if j == len(singleLetterSet) - 1:
                        singleLetterSet.clear()
                else:
                    testingSet.append(singleLetterSet[j])
                    if j == len(singleLetterSet) - 1:
                        singleLetterSet.clear()
    return trainingSet, testingSet
