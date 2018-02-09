from glob import glob
from skimage.measure import block_reduce
from skimage.io import imread
import numpy as np
import re
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
datagen = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.05,
        height_shift_range=0.05,
        shear_range=0.05,
        zoom_range=0.1,
        horizontal_flip=False,
        fill_mode='nearest')
# load the data
negative_paths = glob('SMILEsmileD-master/SMILEs/negatives/negatives7/*.jpg')
positive_paths = glob('SMILEsmileD-master/SMILEs/positives/positives7/*.jpg')

examples = [(path, 0) for path in negative_paths] + [(path, 1) for path in positive_paths]

def generateImages(examples, block_size=2):
    train_X = [] # pixels
    train_y = [] # labels
    test_X = []
    test_y = []
    for path, label in examples:
       num = re.search(r'([0-9]+)\.jpg', path)
       fname = num.group(1)
       img_id = int(fname)
       # read the images
       img = load_img(path)
       img = img_to_array(img)
       img = img.reshape((1,) + img.shape) 
       # scale down the images
       #img = block_reduce(img, block_size=(block_size, block_size), func=np.mean)
       lbldir = "positive" if label else "negative"
       prefix = ("%d" % (img_id+1000000))

       if img_id > 2000:
         i = 0
         for batch in datagen.flow(img, batch_size=1, save_to_dir=('genimages/' +lbldir ), save_prefix=prefix, save_format='jpg'):
           i += 1
           if i > 3:
             break  # otherwise the generator would loop indefinitely
generateImages(examples)
