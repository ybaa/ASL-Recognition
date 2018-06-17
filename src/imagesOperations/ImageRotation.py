from skimage import transform, io
from src.imagesOperations.ImagesCollectionLoader import LoadCollectionFromDir

def RotateImages(dirName, angle):
    images = LoadCollectionFromDir(dirName)
    i = 0;
    for image, file in zip(images, images.files):
       image = transform.rotate(image,angle)
       filename = "%s/%s_Rotated%s_%s.jpg"%(dirName,file[11],angle,i) # file[11] + 'Rotated' + angle + "_" +i+".jpg"
       io.imsave(filename, image)
       i += 1

    print('DONE')
