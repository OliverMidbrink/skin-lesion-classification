import tensorflow as tf
import pandas as pd
from glob import glob
import numpy as np
from PIL import Image
import random
import os
import sys
from tflearn.data_utils import build_hdf5_image_dataset

base_data_dir = 'skin-cancer-mnist-ham10000'

imageid_path_dict = {os.path.splitext(os.path.basename(x))[0]: x
                  for x in glob(os.path.join(base_data_dir, '*', '*.jpg'))}


lesion_type_dict = {
    'nv' : 'Melanocytic nevi',
    'mel' : 'melanoma',
    'bkl' : 'Benign keratosis-like lesions',
    'bcc' : 'Basal cell carcinoma',
    'akiec' : 'Actinic keratoses',
    'vasc' : 'Vascular lesions',
    'df' : 'Dermatofibroma'
}

lesion_id_dict = {
    '4' : 'Melanocytic nevi',
    '6' : 'melanoma',
    '2' : 'Benign keratosis-like lesions',
    '1' : 'Basal cell carcinoma',
    '0' : 'Actinic keratoses',
    '5' : 'Vascular lesions',
    '3' : 'Dermatofibroma'
}

tile_df = pd.read_csv(os.path.join(base_data_dir, 'HAM10000_metadata.csv'))
tile_df['path'] = tile_df["image_id"].map(imageid_path_dict.get)
tile_df['cell_type'] = tile_df['dx'].map(lesion_type_dict.get)
tile_df['cell_type_idx'] = pd.Categorical(tile_df['cell_type']).codes


rows = [tile_df['path'][x] + '\t' + str(tile_df['cell_type_idx'][x]) + '\n' for x in range(len(tile_df['path']))]
random.shuffle(rows)


train_split = 0.9
split_index = int(len(rows) * train_split)
train = rows[0:split_index]
val = rows[split_index:len(rows)]

#print(len(val))
#print(len(train))

print(tile_df[['cell_type_idx', 'cell_type']].sort_values('cell_type_idx').drop_duplicates())

sys.exit()
with open('path_train.data', 'w') as pf:
    pf.writelines(train)
with open('path_val.data', 'w') as pf:
    pf.writelines(val)


build_hdf5_image_dataset('path_train.data',
                         image_shape=(224, 224),
                         mode='file',
                         output_path='train.h5',
                         categorical_labels=True,
                         normalize=True,
                         grayscale=False,
                         )

build_hdf5_image_dataset('path_val.data',
                         image_shape=(224, 224),
                         mode='file',
                         output_path='val.h5',
                         categorical_labels=True,
                         normalize=True,
                         grayscale=False,
                         )
