import os, os.path, sys, platform, time, random, json
from keras.models import load_model
from skimage.transform import resize
import numpy as np
from PIL import Image
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import h5py
import cv2

mac_root_path = '/Users/oliver/Downloads/ISIC-Archive-Downloader-master/Data/'
windows_root_path = 'C:\\Users\\olive\\Desktop\\ISIC-Archive-Downloader-master\\Data\\'
root_path = windows_root_path if platform.system() == 'Windows' else mac_root_path

seg_path = os.path.join(root_path, 'Segmentation')
img_path = os.path.join(root_path, 'Images')
desc_path = os.path.join(root_path, 'Descriptions')
crop_path = os.path.join(root_path, 'Crop')
if not os.path.exists(crop_path):
	os.makedirs(os.path.join(root_path,'Crop'))

model = load_model('best_model_so_far.h5')

desc_names = os.listdir(desc_path)
random.shuffle(desc_names)

data = []


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


def get_bbox(img):
	org_shape = np.shape(img)

	img_224 = cv2.resize(np.asarray(img), (224, 224))

	bbox = model.predict(np.expand_dims(img_224, axis=0))[0]
	bbox[0]+=-10
	bbox[1]+=-10
	bbox[2]+=10
	bbox[3]+=10

	for i in range(len(bbox)):
		bbox[i] = max(1, min(223, bbox[i]))
	#print(bbox)

	#cv2.rectangle(img_224, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (200,0,200), 2)
	#cv2.imshow('img', img_224)
	#k = cv2.waitKey(0)
	#if k == 27:
	#	sys.exit()

	x_scale = org_shape[1]/224
	y_scale = org_shape[0]/224
	bbox_scaled = int(round(bbox[0]*x_scale)), int(round(bbox[1]*y_scale)), int(round(bbox[2]*x_scale)), int(round(bbox[3]*y_scale))

	return bbox_scaled


iterated = 0
start_time = time.time()

def crop_it():
	for x in desc_names:
		iterated+=1
		try:
			save_path = os.path.join(crop_path, x + '_cropped.jpeg')
			if os.path.isfile(save_path):
				print('{} already exists. Skipping'.format(x))
				continue
			img_file_name = os.path.join(img_path, x + '.jpeg')
			if not os.path.isfile(img_file_name):
				continue	# Skip because the image does not exist
			img = Image.open(img_file_name)
			try:
				img.load()
			except:
				print('File {} experienced OS error, truncated file'.format(x))
			img = np.asarray(img)

			bbox = None
			label = None


			desc_file_name = os.path.join(desc_path, x)
			seg_file_name = [p for p in os.listdir(seg_path) if x == p[:12]]

			if len(seg_file_name) > 0:		# If there exists a segmentation mask
				seg = Image.open(os.path.join(seg_path, seg_file_name[0]))
				seg.load()

				a = np.where(np.asarray(seg) != 0)
				try:
					bbox = np.min(a[1]), np.min(a[0]), np.max(a[1]), np.max(a[0])	#x, y, x+w,y+h
				except:
					print('Segmentation for {} could not be scanned. Predicting instead.'.format(x))
					bbox = get_bbox(img)
			else:
				bbox = get_bbox(img)

			if bbox == None:
				print('bbox error: ', x)
				continue

			img_cropped224 = cv2.resize(img[bbox[1]:bbox[3], bbox[0]:bbox[2]], (224, 224))
			

			with open(desc_file_name, 'r') as f:
				desc_file = json.load(f)
				try:
					label = desc_file["meta"]["clinical"]["diagnosis"]
				except:
					print('KeyError for diagnosis in {} desc file'.format(x))
				
			if label == None:
				print('label error: ', x)
				continue

			Image.fromarray(img_cropped224).save(save_path)

			if iterated%20 == 0 and len(data) > 0:
				eta = (time.time()-start_time)/(float(iterated)/23907.0)/60
				print('Iterated {}. There are {} items left. Progress {}%. Time {}. ETA: {}'.format(iterated, 23907-iterated, iterated/239.07, time.time()-start_time, eta))

			if iterated%100 and len(data) > 0:
				ids, addrs, labels = zip(*data)
				df = pd.DataFrame({'ids':ids, 'addrs':addrs, 'labels':labels}, columns=['ids', 'addrs', 'labels'])
				df.to_pickle(os.path.join(root_path, 'classifier_data.pkl'))

			data.append((x, save_path, label))

		except KeyboardInterrupt:
			sys.exit()
		except Exception as e:
			print(e)
		except:
			print('Other')
		# Output 224 image zoomed in on lesion

	df = pd.DataFrame({'ids':ids, 'addrs':addrs, 'labels':labels}, columns=['ids', 'addrs', 'labels'])
	df.to_pickle(os.path.join(root_path, 'classifier_data.pkl'))


def create_hdf5():
	df = pd.read_pickle(os.path.join(root_path, 'classifier_data.pkl'))
	addrs = df['addrs']
	addrs = [str(row) for row in addrs]
	df['label_idx'] = pd.Categorical(df['labels']).codes
	labels = df['label_idx']
	labels = [int(row) for row in labels]


	c = list(zip(addrs, labels))
	random.shuffle(c)
	addrs, labels = zip(*c)

	counter = [0]*18
	percent = [0.0]*18
	for i in range(len(addrs)):
		counter[labels[i]]+=1
		percent[labels[i]]+=1.0/len(addrs)

	print(counter)
	print(percent)

	sys.exit()
	#plt.show()
	#dat = df[['label_idx', 'labels']].sort_values(['label_idx']).drop_duplicates()
	#for i in range(len(dat['label_idx'])):
	#	print('{} : "{}",'.format(dat['label_idx'].iloc[i], dat['labels'].iloc[i]))
	#sys.exit()
	#test
	
	train_addrs = addrs[0:int(0.6*len(addrs))]
	train_labels = labels[0:int(0.6*len(labels))]

	val_addrs = addrs[int(0.6*len(addrs)):int(0.8*len(addrs))]
	val_labels = labels[int(0.6*len(labels)):int(0.8*len(labels))]

	test_addrs = addrs[int(0.8*len(addrs)):]
	test_labels = labels[int(0.8*len(labels)):]





	hdf5_file = h5py.File(os.path.join(root_path, 'classifier_data.h5'))


	train_shape = (len(train_addrs), 224, 224, 3)
	val_shape = (len(val_addrs), 224, 224, 3)
	test_shape = (len(test_addrs), 224, 224, 3)

	hdf5_file.create_dataset("train_img", train_shape, np.uint8)
	hdf5_file.create_dataset("val_img", val_shape, np.uint8)
	hdf5_file.create_dataset("test_img", test_shape, np.uint8)
	hdf5_file.create_dataset("train_mean", train_shape[1:], np.float32)


	hdf5_file.create_dataset("train_labels", (len(train_addrs),), np.int8)
	hdf5_file["train_labels"][...] = train_labels

	hdf5_file.create_dataset("val_labels", (len(val_addrs),), np.int8)
	hdf5_file["val_labels"][...] = val_labels

	hdf5_file.create_dataset("test_labels", (len(test_addrs),), np.int8)
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
	    #img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
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
	    #img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
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
	    #img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
	    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

	    # add any image pre-processing here

	    # save the image
	    hdf5_file["test_img"][i, ...] = img[None]


	hdf5_file["train_mean"][...] = mean
	hdf5_file.close()






create_hdf5()

