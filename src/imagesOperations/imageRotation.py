from skimage import transform, io
from src.imagesOperations.imagesCollectionLoader import load_collection_from_dir

def rotate_image(dir_name, angle):
    images = load_collection_from_dir(dir_name)
    i = 0;
    for image, file in zip(images, images.files):
       image = transform.rotate(image,angle)

       filename = file.rpartition('/')
       filename = filename[len(filename)-1]
       letter = filename[0]
       
       exportFilename = "%s/%s_Rotated%s_%s.jpg"%(dir_name,letter,angle,i) # file[11] + 'Rotated' + angle + "_" +i+".jpg"
       io.imsave(exportFilename, image)
       i += 1

    print('DONE')
