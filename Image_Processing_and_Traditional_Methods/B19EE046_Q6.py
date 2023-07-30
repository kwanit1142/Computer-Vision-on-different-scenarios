import pytesseract
import cv2
import numpy as np
import tarfile
import PIL
from matplotlib import pyplot as plt
import random
import os
from sklearn.neighbors import KNeighborsClassifier as KNC
from sklearn.svm import SVC
from sklearn.metrics import classification_report as cr
import math
import easyocr

def matrix(dir,mode='default'):
  mat=[]
  for name in os.listdir(dir):
    if mode!='default':
      img = cv2.imread(dir+'/'+name,0)
      img[img!=0] = 1
      mat_sum = np.sum(img,axis=1)
      mat.append(mat_sum)
    else:
      img = cv2.imread(dir+'/'+name)
      mat.append(np.reshape(img,-1))
  return np.array(mat)

src_dir = '/content/mnist_png/mnist_png.tar.gz'
file = tarfile.open('/content/mnist_png/mnist_png.tar.gz')
file.extractall('/content/mnist_png/mnist_png_images')
file.close()

dir_0 = '/content/mnist_png/mnist_png_images/mnist_png/training/0/1.png'
dir_1 = '/content/mnist_png/mnist_png_images/mnist_png/training/1/10006.png'

img = cv2.imread(dir_0)
print("Original Image:",img.shape)
cv2.imshow("0",img)

img1 = cv2.imread(dir_1)
print("Original Image:",img1.shape)
cv2.imshow("1",img1)

zero_dir = '/content/mnist_png/mnist_png_images/mnist_png/training/0'
one_dir = '/content/mnist_png/mnist_png_images/mnist_png/training/1'
zero_matrix = matrix(zero_dir)
one_matrix = matrix(one_dir)
zero_labels = np.array([0]*zero_matrix.shape[0])
one_labels = np.array([1]*one_matrix.shape[0])

zero_dirtest = '/content/mnist_png/mnist_png_images/mnist_png/testing/0'
one_dirtest = '/content/mnist_png/mnist_png_images/mnist_png/testing/1'
zero_test = matrix(zero_dirtest)
one_test = matrix(one_dirtest)
zero_labels_test = np.array([0]*zero_test.shape[0])
one_labels_test = np.array([1]*one_test.shape[0])

train_matrix = np.concatenate((zero_matrix,one_matrix),axis=0)
train_labels = np.concatenate((zero_labels,one_labels),axis=0)
shuffled_indices = np.random.permutation(train_matrix.shape[0])
train_matrix=train_matrix[shuffled_indices]
train_labels=train_labels[shuffled_indices]

test_matrix = np.concatenate((zero_test,one_test),axis=0)
test_labels = np.concatenate((zero_labels_test,one_labels_test),axis=0)
shuffled_indices_test = np.random.permutation(test_matrix.shape[0])
test_matrix=test_matrix[shuffled_indices_test]
test_labels=test_labels[shuffled_indices_test]

model_knn = KNC(n_neighbors=5,n_jobs=-1)
model_knn.fit(train_matrix,train_labels)
test_pred = model_knn.predict(test_matrix)

model_svm = SVC(kernel='poly')
model_svm.fit(train_matrix,train_labels)
test_pred1 = model_svm.predict(test_matrix)

print(cr(test_labels,test_pred))
print(cr(test_labels,test_pred1))

zero_matrix1 = matrix(zero_dir,'horizontal')
one_matrix1 = matrix(one_dir,'horizontal')
zero_labels1 = np.array([0]*zero_matrix1.shape[0])
one_labels1 = np.array([1]*one_matrix1.shape[0])

zero_test1 = matrix(zero_dirtest,'horizontal')
one_test1 = matrix(one_dirtest,'horizontal')
zero_labels_test1 = np.array([0]*zero_test1.shape[0])
one_labels_test1 = np.array([1]*one_test1.shape[0])

train_matrix1 = np.concatenate((zero_matrix1,one_matrix1),axis=0)
train_labels1 = np.concatenate((zero_labels1,one_labels1),axis=0)
shuffled_indices1 = np.random.permutation(train_matrix1.shape[0])
train_matrix1=train_matrix1[shuffled_indices1]
train_labels1=train_labels1[shuffled_indices1]

test_matrix1 = np.concatenate((zero_test1,one_test1),axis=0)
test_labels1 = np.concatenate((zero_labels_test1,one_labels_test1),axis=0)
shuffled_indices_test1 = np.random.permutation(test_matrix1.shape[0])
test_matrix1=test_matrix1[shuffled_indices_test1]
test_labels1=test_labels1[shuffled_indices_test1]

model_knn1 = KNC(n_neighbors=10,n_jobs=-1)
model_knn1.fit(train_matrix1,train_labels1)
test_pred11 = model_knn1.predict(test_matrix1)

model_svm1 = SVC(kernel='linear')
model_svm1.fit(train_matrix1,train_labels1)
test_pred12 = model_svm1.predict(test_matrix1)

print(cr(test_labels1,test_pred11))
print(cr(test_labels1,test_pred12))