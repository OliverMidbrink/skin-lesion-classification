import os, os.path, sys, platform, time, random, keras
import numpy as np
import h5py
import cv2
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

from keras.utils import np_utils
from keras import metrics
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Dropout, Conv2D, Flatten, MaxPooling2D, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam, Nadam
from keras.backend.tensorflow_backend import set_session
from tensorflow.python.keras.utils.data_utils import Sequence
import tensorflow as tf
from keras.applications import VGG16
from keras.callbacks.callbacks import ReduceLROnPlateau


if platform.system() == 'Windows':
	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
	sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
	set_session(sess)


#							------------- Structure Data ---------------
mac_root_path = '/Users/oliver/Downloads/ISIC-Archive-Downloader-master/Data/'
windows_root_path = 'C:\\Users\\olive\\Desktop\\ISIC-Archive-Downloader-master\\Data\\'
root_path = windows_root_path if platform.system() == 'Windows' else mac_root_path


h5 = h5py.File(os.path.join(root_path, 'classifier_data.h5'), 'r')
x_train = h5['train_img']
y_train = np_utils.to_categorical(h5['train_labels'])

x_val = h5['val_img']
y_val = np_utils.to_categorical(h5['val_labels'])
x_test = h5['test_img']
y_test = np_utils.to_categorical(h5['test_labels'])


diagnosis_dict = {
	0 : "actinic keratosis",
	1 : "angiofibroma or fibrous papule",
	2 : "angioma",
	3 : "atypical melanocytic proliferation",
	4 : "basal cell carcinoma",
	5 : "dermatofibroma",
	6 : "lentigo NOS",
	7 : "lentigo simplex",
	8 : "lichenoid keratosis",
	9 : "melanoma",
	10 : "nevus",
	11 : "other",
	12 : "pigmented benign keratosis",
	13 : "scar",
	14 : "seborrheic keratosis",
	15 : "solar lentigo",
	16 : "squamous cell carcinoma",
	17 : "vascular lesion"
}

weight_dict = {
	0 : 17,
	1 : 30.1,
	2 : 20,
	3 : 80,
	4 : 5,
	5 : 7,
	6 : 30,
	7 : 60,
	8 : 20,
	9 : 2.8,
	10 : 0.09,
	11 : 104,
	12 : 1,
	13 : 0,
	14 : 2.8,
	15 : 17,
	16 : 12,
	17 : 11
} # Dangerous types are increased by 70%, non dangerous types decrased by 40%.  Very small datacategories have been heavily reduced (0, 8) from 840 to 30


def create_categori_map(y_in):			#creates a map of all data, axis 0 represent categories, 
	out_size = [0] * len(y_in[0])		# each categori element contains list of indices to data 
	indices = [np.where(lab == 1)[0][0] for lab in np.array(y_in)]	# of that categori
										# For instance 
	for lab_i in range(len(indices)):	# [ idxs of cat 0 --> [3466, 132], cat 1: [867,1,2], [0] ]		
		out_size[indices[lab_i]]+=1

	out_map = [[0] * elem for elem in out_size]
	counter = [0] * len(y_in[0])
	for i in range(len(indices)):
		lab = indices[i]	#this is the label for image id = i
		out_map[lab][counter[lab]] = i
		counter[lab]+=1

	return out_map
	

train_map = create_categori_map(y_train)
val_map = create_categori_map(y_val)
test_map = create_categori_map(y_test)


y_train_int = np.argmax(y_train, axis=1)
class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(y_train_int),
                                                 y_train_int)

class_weights = np.insert(class_weights, 13, 0)
class_weight_dict = dict(enumerate(class_weights))


#for i in range(len(train_map)):
#	print(len(train_map[i]), "\t\t", class_weight_dict[i], "\t\t", i, "\t\t", diagnosis_dict[i])


#							------------- Define Model ---------------
model = Sequential()

vgg_conv = VGG16(weights = 'imagenet', include_top=False, input_shape=(224,224,3))

for layer in vgg_conv.layers[:-12]:
	layer.trainable = False

model.add(vgg_conv)
model.add(Flatten())
model.add(Dense(18, activation = 'softmax'))

#opt = keras.optimizers.Adadelta(lr=0.00001, rho=0.95)
reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', factor=0.5, verbose=1, patience=4, min_lr = 0.000001)
opt = Adam(lr=0.0002)
model.compile(loss = 'categorical_crossentropy', 
			optimizer = opt,
			metrics = [metrics.categorical_accuracy])


# ----- Exeption Model ---------
xception_base = keras.applications.xception.Xception(include_top=False, weights='imagenet', input_shape=(224,224,3))


for layer in xception_base.layers:
	layer.trainable = False

x = xception_base.output
x = GlobalAveragePooling2D() (x)
x = Dense(250, activation='relu') (x)
x = Dropout(0.5) (x)
x = Dense(120, activation='relu') (x)
predictions = Dense(18, activation = 'softmax') (x)

xception_model = Model(xception_base.input, predictions)
xception_model.compile(loss = 'categorical_crossentropy', 
			optimizer=Nadam(lr=0.0001),
			metrics = [metrics.categorical_accuracy])


class CustomSequence(Sequence):
	def __init__(self, x_set, y_set, cat_map, batch_size, augmentations):
		self.x, self.y = x_set, y_set
		self.cat_map = cat_map
		self.batch_size = batch_size
		self.augmentations = augmentations

	def __len__(self):
		return int(np.ceil(len(self.x) / float(self.batch_size)))

	def __getitem__(self, idx):
		while True:
			batch_x = []
			batch_y = []

			dim = [len(elem) for elem in self.cat_map]
			max_categori_length = max(dim)
			
			for i in range(max_categori_length):
				for cat in range(len(self.cat_map)):
					if dim[cat] == 0:
						continue
					idx_append = self.cat_map[cat][i%dim[cat]]
					
					batch_x.append(self.x[idx_append])
					batch_y.append(self.y[idx_append])
					
					if len(batch_x) == self.batch_size:
						aug_x = [self.augmentations(elem_x) for elem_x in batch_x]
						z = list(zip(aug_x, batch_y))
						random.shuffle(z)
						aug_x, batch_y = zip(*z)
						yield np.array(aug_x), np.array(batch_y)
						batch_x = []
						batch_y = []

def augment(x_in):
	x_out = x_in.copy()

	if True:	#Should it random flip?
		if random.randint(0,1):		#Random flip up-down
			x_out = np.flip(x_out, axis=0)

		if random.randint(0,1):		#Random flip left-right
			x_out = np.flip(x_out, axis=1)
	
	# Change contrast (factor should be float)
	factor = float(2.7 * random.random() + 0.1)
	np.clip(128 + factor * (x_out - 128.0), 0 , 255, out = x_out).astype(np.uint8)

	

	# Random Shift brightnesss
	max_shift = 65.0		#50 for 600epoch
	np.clip(x_out + float((2 * random.random() - 1) * max_shift), 0, 255, out = x_out).astype(np.uint8)

	dx = random.randint(-150, 150)
	dy = random.randint(-150, 150)

	x_out = np.roll(x_out, dy, axis=0)
	x_out = np.roll(x_out, dx, axis=1)
	if dy>0:
		x_out[:dy, :] = 0
	elif dy<0:
		x_out[dy:, :] = 0
	if dx>0:
		x_out[:, :dx] = 0
	elif dx<0:
		x_out[:, dx:] = 0
	
	return x_out


batch_size = 2 * len(train_map)
epochs = 200

image_gen = CustomSequence(x_train, y_train, train_map, batch_size, augment)
val_image_gen = CustomSequence(x_val, y_val, val_map, batch_size, augment)


filepath = "checkE\\weights-improvement-{epoch:02d}-{val_categorical_accuracy:.2f}.hdf5"
check = keras.callbacks.callbacks.ModelCheckpoint(filepath, monitor='val_categorical_accuracy', verbose=1, save_best_only=True, save_weights_only=False, mode='max', period=1)

file = 'check, even distribution\\weights-improvement-03-0.64.hdf5'
if os.path.isfile(file):
	print("----------- FILE FOUND; LOADING {} !! -----------------".format(file))
	model = load_model(file)

#model.fit_generator(next(iter(image_gen)), validation_data = (x_val, y_val), steps_per_epoch = len(image_gen.x)/batch_size, epochs = epochs, callbacks=[reduce_lr, check])
#model.save("checkE\\EXP2.hdf5")
#steps_per_epoch = len(image_gen.x)/batch_size


#model.fit(x = x_train, y = y_train, validation_data = (x_val, y_val), batch_size = batch_size, shuffle='batch', epochs = epochs, callbacks=[cb, check])
#model.save_weights("organic-1.hdf5") # No aug


train_gen = ImageDataGenerator(rotation_range=180,
							brightness_range=(-50,50),
							zoom_range=0.6,
							horizontal_flip=True,
							vertical_flip=True,
							width_shift_range = 0.6,
							height_shift_range = 0.6,
							preprocessing_function=augment)

#model.fit_generator(train_gen.flow(x_train, y_train, batch_size=batch_size), steps_per_epoch = len(x_train)/batch_size,
#							 validation_data = (x_val, y_val), epochs = epochs, callbacks=[check, reduce_lr], class_weight = weight_dict)

#model.save('checkE\\adjWeights70epoch12unfrozen.hdf5')
#xception_model.fit(x = x_train, y = y_train, validation_data = (x_val, y_val), batch_size = batch_size, shuffle='batch', epochs = epochs, callbacks=[check])
#xception_model.fit_generator(train_gen.flow(x_train, y_train, batch_size=batch_size), steps_per_epoch = len(x_train)/batch_size,
#							 validation_data = (x_val, y_val), epochs = epochs, callbacks=[check, cb], class_weight = class_weight_dict)


'''
losses = []
accuracys = []
for cat in range(len(val_map)):
	if len(val_map[cat]) == 0:
		losses.append(0)
		accuracys.append(0)
		continue

	batch_x = []
	batch_y = []

	for i in range(len(val_map[cat])):
		data_index = val_map[cat][i]
		batch_x.append(x_val[data_index])
		batch_y.append(y_val[data_index])

	result = model.evaluate(np.array(batch_x), np.array(batch_y), batch_size)
	losses.append(round(result[0], 4))
	accuracys.append(round(result[1], 4))

print(accuracys)
print([len(elem) for elem in val_map])
print('\n\n\n\n\n\n')
for i in accuracys:
	print(i)
print('\n\n')
for i in val_map:
	print(len(i))
'''


# EVALUATE WITH TEST DATA, don't overdo
def evaluate(eval_model, cat_map, x, y):
	print("---------------    EVALUATION    -----------------")
	losses = [0] * len(cat_map)
	accuracys = [0] * len(cat_map)
	for cat_number in range(len(cat_map)):		# Iterate from cat. 1 through category 18 (in this case)
		if len(cat_map[cat_number]) == 0:		# If there is no data in this category, skip
			continue

		batch_x = []
		batch_y = []

		for i in range(len(cat_map[cat_number])):			# Iterate through all indexes included in cat_map's category number cat (see above for loop)
			data_index = cat_map[cat_number][i]			
			batch_x.append(x[data_index])
			batch_y.append(y[data_index])

		result = model.evaluate(np.array(batch_x), np.array(batch_y), batch_size)
		losses[cat_number] = round(result[0], 4)
		accuracys[cat_number] = round(result[1], 4)

	print(accuracys)
	print([len(elem) for elem in cat_map])
	print('\n\n\n\n\n\n')
	for i in accuracys:
		print(i)
	print('\n\n')
	for i in cat_map:
		print(len(i))


evaluate(model, test_map, x_test, y_test)


y_pred = np.append(np.argmax(model.predict(x_test), axis = 1), range(18))
cm = confusion_matrix(np.append(np.argmax(y_test, axis=1), range(18)), y_pred)

df_cm = pd.DataFrame(cm, diagnosis_dict.values(), diagnosis_dict.values())

sn.set(font_scale=1)

plt.figure(figsize = (10,7))
sn.heatmap(df_cm, annot=True, annot_kws={"size": 6})
plt.show()



#info here https://stackoverflow.com/questions/35572000/how-can-i-plot-a-confusion-matrix
#evaluate(model, test_map, x_test, y_test)


# Test interactively
'''
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
'''
