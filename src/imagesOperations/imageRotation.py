from skimage import transform, io
from imagesOperations.imagesCollectionLoader import load_collection_from_dir

def rotate_image(dir_name, angle):
    images = load_collection_from_dir(dir_name)
    i = 0;
    for image, file in zip(images, images.files):
       image = transform.rotate(image,angle)
       filename = "%s/%s_Rotated%s_%s.jpg"%(dir_name,file[11],angle,i) # file[11] + 'Rotated' + angle + "_" +i+".jpg"
       io.imsave(filename, image)
       i += 1

    print('DONE')
