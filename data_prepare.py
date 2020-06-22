import scipy.io as sio
import numpy as np
import tqdm
import tensorflow as tf

def extract_gtsamples(X, y):
  
  from sklearn.utils import shuffle
  import tqdm
  shapeX = np.shape(X)

  index = np.empty([0,3], dtype = 'int')

  for k in tqdm.tqdm(range(1,np.size(np.unique(y)))):
    for i in range(shapeX[0]):
      for j in range(shapeX[1]):
        if y[i,j] == k:
          index = np.append(index,np.expand_dims(np.array([k,i,j]),0),0)

  VecX = np.empty([index.shape[0],shapeX[2]])
  Y = np.empty([index.shape[0]])

  for i in range(index.shape[0]):
    p = index[i,1]
    q = index[i,2]
    VecX[i,:] = X[p,q,:]
    Y[i] = index[i,0]

  VecX, Y = shuffle(VecX, Y)
  return VecX, Y-1

data = sio.loadmat('/data/Salinas_corrected.mat')['salinas_corrected']
label = sio.loadmat('/data/Salinas_gt.mat')['salinas_gt']

feats_norm = np.empty([512,217,204], dtype = 'float32')
for i in tqdm.tqdm(range(204)):
  feats_norm[:,:,i] = data[:,:,i] - np.min(data[:,:,i])
  feats_norm[:,:,i] = feats_norm[:,:,i]/np.max(feats_norm[:,:,i])

train_test_vec, train_test_labels = extract_gtsamples(feats_norm, label)

np.shape(train_test_labels)

from sklearn.model_selection import StratifiedShuffleSplit
s3 = StratifiedShuffleSplit(n_splits=1, test_size=0.95, random_state=0)
s3.get_n_splits(train_test_vec, train_test_labels)

for train_index, test_index in s3.split(train_test_vec, train_test_labels):
   train_vec, test_vec = train_test_vec[train_index], train_test_vec[test_index]
   train_labels, test_labels = train_test_labels[train_index], train_test_labels[test_index]

np.save('/data/train_vec_salinas',train_vec)
np.save('data/test_vec_salinas',test_vec)
np.save('data/train_labels_salinas',train_labels)
np.save('data/test_labels_salinas',test_labels)