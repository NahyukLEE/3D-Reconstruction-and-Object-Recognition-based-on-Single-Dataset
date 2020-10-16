from numpy import expand_dims
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
import cv2

# load the image
img = load_img('frame5.jpg')

# convert to numpy array
data = img_to_array(img)

# expand dimension to one sample
samples = expand_dims(data, 0)

# PARAMETER SETTING
SHIFT_RANGE = 0.25 # horizontal_shift, vertical_shift에 적용



def horizontal_shift():
	# create image data augmentation generator
	datagen = ImageDataGenerator(width_shift_range=SHIFT_RANGE)
	# prepare iterator
	it = datagen.flow(samples, batch_size=1)

	for i in range(9):
		# generate batch of images
		batch = it.next()
		# convert to unsigned integers for viewing
		image = batch[0].astype('uint8')

		img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
		cv2.imshow('img', img)

def vertical_shift():
	# create image data augmentation generator
	datagen = ImageDataGenerator(height_shift_range=SHIFT_RANGE)
	# prepare iterator
	it = datagen.flow(samples, batch_size=1)

	for i in range(9):
		# generate batch of images
		batch = it.next()
		# convert to unsigned integers for viewing
		image = batch[0].astype('uint8')

		img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
		cv2.imshow('img', img)
		cv2.waitKey()
		cv2.destroyAllWindows()

def

