import os, os.path, sys, platform, time, random
import numpy as np
import h5py
import cv2

from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Conv2D, Flatten, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam, SGD
from keras.backend.tensorflow_backend import set_session
from tensorflow.python.keras.utils.data_utils import Sequence
import tensorflow as tf

#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
#sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
#set_session(sess)

batch_size = 50

mac_root_path = '/Users/oliver/Downloads/ISIC-Archive-Downloader-master/Data/'
windows_root_path = 'C:\\Users\\olive\\Desktop\\ISIC-Archive-Downloader-master\\Data\\'
root_path = windows_root_path if platform.system() == 'Windows' else mac_root_path


h5 = h5py.File(os.path.join(root_path, 'loc_data.h5'), 'r')
x_train = h5['train_img']
y_train = h5['train_labels']
x_val = h5['val_img']
y_val = h5['val_labels']
x_test = h5['test_img']
y_test = h5['test_labels']


#		Make bounding boxes 10 px smaller
y_train = np.array([[y[0]+10, y[1]+10, y[2]-10, y[3]-10] for y in y_train])
y_val = np.array([[y[0]+10, y[1]+10, y[2]-10, y[3]-10] for y in y_val])

#		Select which data to use
#x_total = np.concatenate((x_train, x_val), axis = 0)
#y_total = np.concatenate((y_train, y_val), axis = 0)	


model = Sequential()

model.add(Conv2D(64, kernel_size=3, activation = 'relu', input_shape = (224, 224, 3)))
model.add(MaxPooling2D())
model.add(Conv2D(64, kernel_size=3, activation = 'relu'))
model.add(MaxPooling2D(pool_size=(4,4)))
model.add(Conv2D(64, kernel_size=3, activation = 'relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(128, activation = 'relu'))
model.add(Dense(4, activation = 'linear', use_bias=True))

model2 = Sequential()

model2.add(Conv2D(64, kernel_size=3, activation = 'relu', input_shape = (224, 224, 3)))
model2.add(MaxPooling2D())
model2.add(Conv2D(64, kernel_size=3, activation = 'relu'))
model2.add(MaxPooling2D())
model2.add(Conv2D(64, kernel_size=3, activation = 'relu'))
model2.add(MaxPooling2D())
model2.add(Flatten())
model2.add(Dense(128, activation = 'relu'))
model2.add(Dense(4, activation = 'relu', use_bias=False))

#opt = keras.optimizers.Adadelta(lr=0.00001, rho=0.95)
opt = Adam(lr=0.0001)
opt2 = SGD(lr=0.0001, momentum=0.1, nesterov=False)
model.compile(loss = 'mean_squared_error', optimizer = opt)
model2.compile(loss = 'mean_squared_error', optimizer = opt)

class CustomSequence(Sequence):
	def __init__(self, x_set, y_set, batch_size, augmentations):
		self.x, self.y = x_set, y_set
		self.batch_size = batch_size
		self.augmentations = augmentations

	def __len__(self):
		return int(np.ceil(len(self.x) / float(self.batch_size)))

	def __getitem__(self, idx):
		while True:
			batch_x = []
			batch_y = []

			for i in range(len(self.x)):
				batch_x.append(self.x[i])
				batch_y.append(self.y[i])

				if len(batch_x) == self.batch_size:
					batch_xy = [self.augmentations(batch_x[i], batch_y[i]) for i in range(len(batch_x))]
					aug_x, aug_y = zip(*batch_xy)
					yield np.array(aug_x), np.array(aug_y)
					batch_x = []
					batch_y = []

def augment(x_in, y_in):
	x_out = x_in
	y_out = y_in

	if random.randint(0,1):		#Random flip up-down
		x_out = np.flipud(x_in)
		y_out[1] = 224 - y_in[3]
		y_out[3] = 224 - y_in[1]

	if random.randint(0,1):		#Random flip left-right
		x_out = np.fliplr(x_in)
		y_out[0] = 224 - y_in[2]
		y_out[2] = 224 - y_in[0]


	dx = random.randint(-y_out[0], 224-y_out[2])
	dy = random.randint(-y_out[1], 224-y_out[3])
	#dx = -y_in[0]
	#dy = -y_in[1]

	y_out[0]+=dx
	y_out[2]+=dx
	y_out[1]+=dy
	y_out[3]+=dy

	mean_r = np.mean(x_out)

	x_out = np.roll(x_out, dy, axis=0)
	x_out = np.roll(x_out, dx, axis=1)
	if dy>0:
		x_out[:dy, :] = mean_r
	elif dy<0:
		x_out[dy:, :] = mean_r
	if dx>0:
		x_out[:, :dx] = mean_r
	elif dx<0:
		x_out[:, dx:] = mean_r

	return x_out, y_out

image_gen = CustomSequence(x_train, y_train, batch_size, augment)
#image_gen.fit(x_train)
'''
x_b, y_b = next(iter(image_gen.__getitem__(0)))
for i in range(len(x_b)):
	img_normalized = np.asarray(x_b[i], dtype=np.uint8)
	img_f = np.asarray(img_normalized.copy(), dtype=np.uint8)
	cv2.rectangle(img_normalized, (y_b[i][0], y_b[i][1]), (y_b[i][2], y_b[i][3]), (200,200,200), 2)
	cv2.rectangle(img_normalized, (224 - y_b[i][2], 224 - y_b[i][3]), (224 - y_b[i][0], 224 - y_b[i][1]), (200,0,200), 2)
	cv2.imshow('Img', img_normalized)
	img_normalized = img_f
	img_normalized = np.flipud(img_normalized)
	img_normalized = np.fliplr(img_normalized)
	#cv2.rectangle(img_normalized, (224 - y_b[i][2], 224 - y_b[i][3]), (224 - y_b[i][0], 224 - y_b[i][1]), (200,200,200), 2)
	cv2.imshow('IMG FLIP', img_normalized)
	key = cv2.waitKey(0)
	if key == 27:
		break

cv2.destroyAllWindows()
sys.exit()
'''

file = 'W2-1.h5'
if os.path.isfile(file):
	model2.load_weights(file)
#model.fit(x_train, y_train, validation_data = (x_val, y_val), epochs = 5, batch_size=60, shuffle = 'batch')
model2.fit_generator(next(iter(image_gen)), validation_data = (x_val, y_val), steps_per_epoch = len(image_gen.x)/batch_size, epochs = 15)
# validation_data = (x_val, y_val),
model2.save_weights(file)
#model.save(file)


# Test interactively
if input('Validate') == 'y':
	while True:
		i = random.randint(0, len(x_val)-1)
		img = x_val[i]
		x_pred = np.expand_dims(img, axis=0)
		print('SHAPE {}'.format(x_pred.shape))
		label = model.predict(x_pred)[0]
		print(y_val[i])
		print(label)
		cv2.rectangle(img, (label[0], label[1]), (label[2], label[3]), (200,0,200), 2)
		cv2.rectangle(img, (y_val[i][0], y_val[i][1]), (y_val[i][2], y_val[i][3]), (200,200,200), 2)
		cv2.imshow('',img)

		key = cv2.waitKey(0)
		if key == 27:
			break

cv2.destroyAllWindows()