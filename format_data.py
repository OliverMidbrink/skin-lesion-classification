import os, os.path, sys, platform, time
from skimage.measure import regionprops
from skimage.transform import resize
from PIL import Image
import numpy as np
import matplotlib
import h5py
import cv2
import pandas as pd


mac_root_path = '/Users/oliver/Downloads/ISIC-Archive-Downloader-master/Data/'
windows_root_path = 'C:\\Users\\olive\\Desktop\\ISIC-Archive-Downloader-master\\Data\\'
root_path = windows_root_path if platform.system() == 'Windows' else mac_root_path

seg_path = os.path.join(root_path, 'Segmentation')
img_path = os.path.join(root_path, 'Images')
desc_path = os.path.join(root_path, 'Descriptions')
locator_data_path = os.path.join(root_path, 'Clean_loc')
if not os.path.exists(locator_data_path):
	os.makedirs(locator_data_path)

desc_names = os.listdir(desc_path)
desc_names.sort()
desc_names = desc_names[:]

n_items = len(desc_names)
n_iterated = 1
start_time = time.time()
bad_files = []
labels = []
paths = []
for x in desc_names:
	save_path = os.path.join(locator_data_path, str(x) + '_224x224.jpeg')
	if os.path.isfile(save_path):
		print(x, ' exists')
		continue		# File exists so skip it

	h, m = divmod((time.time()-start_time+1)*n_items/n_iterated/60, 60)
	print('{} hours {:.2f} minutes.\tIterated: {}\t{}'.format(h, m, n_iterated-1, x))

	# process image
	desc_file_name = os.path.join(desc_path, x)
	img_file_name = os.path.join(img_path, x + '.jpeg')
	seg_file_name = [p for p in os.listdir(seg_path) if x == p[:12]]

	if len(seg_file_name)>0: #this code will iterate through all available segmentations
		print('Creating ', x)
		seg = Image.open(os.path.join(seg_path, seg_file_name[0]))
		img = Image.open(img_file_name)
		
		seg.load()
		img.load()

		data_size = (224,224)

		a = np.where(np.asarray(seg) != 0)
		try:
			bbox = np.min(a[1]), np.min(a[0]), np.max(a[1]), np.max(a[0])	#x, y, x+w,x+h
		except:
			print('File {} could not be created, No segmentation data available!!'.format(x))
			bad_files.append(x)
			continue
		x_scale = 224/np.shape(seg)[1]
		y_scale = 224/np.shape(seg)[0]

		x_elem = (resize(np.asarray(img), data_size) * 255).astype('uint8')
		print(x_elem)
		y_elem = int(round(bbox[0]*x_scale)), int(round(bbox[1]*y_scale)), int(round(bbox[2]*x_scale)), int(round(bbox[3]*y_scale))
		#print(y, bbox, np.shape(seg))
		

		#				Demonstrate Bounding box
		cv2.rectangle(x_elem, (y_elem[0], y_elem[1]), (y_elem[2], y_elem[3]), (200,200,200), 2)
		cv2.rectangle(x_elem, (y_elem[0]+10, y_elem[1]+10), (y_elem[2]-10, y_elem[3]-10), (0,0,200), 2)
		#cv2.imshow('',x_elem)
		#cv2.rectangle(seg, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (200,200,200), 2)
		#print(bbox)			
		cv2.imshow('seg', (resize(np.asarray(seg), data_size) * 255).astype('uint8'))
		cv2.imshow('img', x_elem)
		key = cv2.waitKey(0)
		if key == ord('y') and False:
			matplotlib.image.imsave(save_path, x_elem)
			labels.append(y_elem)
			paths.append(save_path)
		elif key == ord('q'):
			#df = pd.DataFrame({'addrs':paths, 'labels':labels}, columns=['addrs', 'labels'])
			#df.to_pickle(os.path.join(root_path, 'clean_loc.pkl'))
			cv2.destroyAllWindows()
			sys.exit()
	n_iterated+=1


print('bad files: \n', bad_files)
