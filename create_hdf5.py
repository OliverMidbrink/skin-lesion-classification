import os, os.path, sys, platform, time, random
from skimage.measure import regionprops
from skimage.transform import resize
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib
import h5py
import cv2

mac_root_path = '/Users/oliver/Downloads/ISIC-Archive-Downloader-master/Data/'
windows_root_path = 'C:\\Users\\olive\\Desktop\\ISIC-Archive-Downloader-master\\Data\\'
root_path = windows_root_path if platform.system() == 'Windows' else mac_root_path

seg_path = os.path.join(root_path, 'Segmentation')
img_path = os.path.join(root_path, 'Images')
desc_path = os.path.join(root_path, 'Descriptions')
locator_data_path = os.path.join(root_path, 'Locator')




if not os.path.isfile(os.path.join(root_path, 'clean_loc.pkl')):
	local_addrs = os.listdir(locator_data_path)
else:
	local_addrs = []
n = len(local_addrs)
random.shuffle(local_addrs)
labels = []
addrs = []


i=0
for addr in local_addrs:
	seg_file_name = [p for p in os.listdir(seg_path) if addr[:12] == p[:12]]
	seg = Image.open(os.path.join(seg_path, seg_file_name[0]))
	seg.load()

	a = np.where(np.asarray(seg) != 0)
	try:
		bbox = np.min(a[1]), np.min(a[0]), np.max(a[1]), np.max(a[0])	#x, y, x+w,x+h
	except:
		print('File {} could not be created, No segmentation data available!!'.format(local_addrs))
		continue

	x_scale = 224/np.shape(seg)[1]
	y_scale = 224/np.shape(seg)[0]

	label = int(round(bbox[0]*x_scale)), int(round(bbox[1]*y_scale)), int(round(bbox[2]*x_scale)), int(round(bbox[3]*y_scale))
	labels.append(label)
	addrs.append(os.path.join(locator_data_path, addr))
	


	i+=1
	if i%20:
		print(n-len(labels), n-len(addrs))
	'''
	img = Image.open(os.path.join(locator_data_path, addr))
	img.load()
	img = np.asarray(img)
	cv2.rectangle(img, (label[0], label[1]), (label[2], label[3]), (200,200,200), 2)
	cv2.imshow('',img)
	cv2.waitKey(0)
	'''

#print('Shuffling data')
#c = list(zip(addrs, labels))
#random.shuffle(c)
#addrs, labels = zip(*c)

print('Saving data')
#df = pd.DataFrame({'addrs':addrs, 'labels':labels}, columns=['addrs', 'labels'])
#df.to_pickle(os.path.join(root_path, 'data.pkl'))
df = pd.read_pickle(os.path.join(root_path, 'clean_loc.pkl'))
addrs = df['addrs']
addrs = [str(row) for row in addrs]
labels = df['labels']
labels = [list(row) for row in labels]
print('Done.')


train_addrs = addrs[0:int(0.6*len(addrs))]
train_labels = labels[0:int(0.6*len(labels))]

val_addrs = addrs[int(0.6*len(addrs)):int(0.8*len(addrs))]
val_labels = labels[int(0.6*len(labels)):int(0.8*len(labels))]

test_addrs = addrs[int(0.8*len(addrs)):]
test_labels = labels[int(0.8*len(labels)):]

train_shape = (len(train_addrs), 224, 224, 3)
val_shape = (len(val_addrs), 224, 224, 3)
test_shape = (len(test_addrs), 224, 224, 3)

hdf5_file = h5py.File(os.path.join(root_path, 'loc_data_clean.h5'))

hdf5_file.create_dataset("train_img", train_shape, np.uint8)
hdf5_file.create_dataset("val_img", val_shape, np.uint8)
hdf5_file.create_dataset("test_img", test_shape, np.uint8)
hdf5_file.create_dataset("train_mean", train_shape[1:], np.float32)

hdf5_file.create_dataset("train_labels", (len(train_addrs), 4), np.int16)
hdf5_file["train_labels"][...] = train_labels
hdf5_file.create_dataset("val_labels", (len(val_addrs), 4), np.int16)
hdf5_file["val_labels"][...] = val_labels
hdf5_file.create_dataset("test_labels", (len(test_addrs), 4), np.int16)
hdf5_file["test_labels"][...] = test_labels

mean = np.zeros(train_shape[1:], np.float32)

for i in range(len(train_addrs)):
    # print how many images are saved every 1000 images
    if i % 1000 == 0 and i > 1:
        print ('Train data: {}/{}'.format(i, len(train_addrs)))

    # read an image and resize to (224, 224)
    # cv2 load images as BGR, convert it to RGB
    addr = train_addrs[i]
    img = cv2.imread(addr)
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # add any image pre-processing here

    # save the image and calculate the mean so far
    hdf5_file["train_img"][i, ...] = img[None]
    mean += img / float(len(train_labels))


for i in range(len(val_addrs)):
    # print how many images are saved every 1000 images
    if i % 1000 == 0 and i > 1:
        print ('Validation data: {}/{}'.format(i, len(val_addrs)))

    # read an image and resize to (224, 224)
    # cv2 load images as BGR, convert it to RGB
    addr = val_addrs[i]
    img = cv2.imread(addr)
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # add any image pre-processing here

    # save the image
    hdf5_file["val_img"][i, ...] = img[None]

for i in range(len(test_addrs)):
    # print how many images are saved every 1000 images
    if i % 1000 == 0 and i > 1:
        print ('Test data: {}/{}'.format(i, len(test_addrs)))

    # read an image and resize to (224, 224)
    # cv2 load images as BGR, convert it to RGB
    addr = test_addrs[i]
    img = cv2.imread(addr)
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # add any image pre-processing here

    # save the image
    hdf5_file["test_img"][i, ...] = img[None]


hdf5_file["train_mean"][...] = mean
hdf5_file.close()