import requests
import h5py
import numpy as np
import zipfile
from tqdm import tqdm
import math

def file_download(filename):
    url = "https://zenodo.org/record/1442704/files/"+filename
    r = requests.get(url, stream=True)
    total_size = int(r.headers.get('content-length', 0))
    block_size = 1024
    wrote = 0 
    with open(filename, 'wb') as f:
    	for data in tqdm(r.iter_content(block_size), total=math.ceil(total_size//block_size) , unit='KB', desc = filename, leave = True):
        	wrote = wrote  + len(data)
        	f.write(data)
    if total_size != 0 and wrote != total_size:
        print("ERROR, something went wrong")
    f.close
def zip_download(filename):
    url = "https://zenodo.org/record/1442708/files/"+filename
    r = requests.get(url, stream=True)
    total_size = int(r.headers.get('content-length', 0))
    block_size = 1024
    wrote = 0 
    with open(filename, 'wb') as f:
    	for data in tqdm(r.iter_content(block_size), total=math.ceil(total_size//block_size) , unit='KB', desc = filename, leave = True):
        	wrote = wrote  + len(data)
        	f.write(data)
    if total_size != 0 and wrote != total_size:
        print("ERROR, something went wrong")
    f.close
    zip_ref = zipfile.ZipFile(filename, 'r')
    zip_ref.extractall(".")
    zip_ref.close()
def load():
    print("LOADING DATASETS AND MODEL WEIGHTS\n")
    file_download("title_abstract_5class.h5")
    file_download("title_abstract_5class_weights.h5")
    file_download("figures_5class.h5")
    file_download("figures_5class_weights.h5")
    file_download("captions_5class.h5")
    file_download("captions_5class_weights.h5")
    file_download("cross.h5")
    file_download("cross_weights.h5")
    file_download("captions_5class_cross.h5")
    file_download("captions_5class_cross_weights.h5")
    file_download("figures_5class_cross.h5")
    file_download("figures_5class_cross_weights.h5")
    file_download("quality5class.h5")
    file_download("qualityMix5class.h5")
    file_download("qualityUni5class.h5")
    zip_download("5class.zip")

def gen_text (h5path, indices,batchSize, shuffle): 
  db = h5py.File(h5path, "r")
  while True:
    if shuffle:
        np.random.shuffle(indices)
    for i in range(0, len(indices), batchSize):
        batch_indices = indices[i:i+batchSize]
        batch_indices.sort()
        
        bx = db["text"][batch_indices,:]
        by = db["labels"][batch_indices,:]

        yield (bx, by)

def gen_images (h5path, indices,batchSize, shuffle):
  db = h5py.File(h5path, "r")
  while True:
    if shuffle:
        np.random.shuffle(indices)
    for i in range(0, len(indices), batchSize):
        batch_indices = indices[i:i+batchSize]
        batch_indices.sort()
        
        bx = db["images"][batch_indices,:,:,:]
        by = db["labels"][batch_indices,:]

        yield (bx, by)

def gen_cross (h5path, indices,batchSize,shuffle):
  db = h5py.File(h5path, "r")
  while True:
    if shuffle:
        np.random.shuffle(indices)
    for i in range(0, len(indices), batchSize):
        batch_indices = indices[i:i+batchSize]
        batch_indices.sort()
        
        bx1 = db["text"][batch_indices,:]
        bx2 = db["images"][batch_indices,:,:,:]
        by = db["labels"][batch_indices,:]

        yield ([bx1, bx2], by)

