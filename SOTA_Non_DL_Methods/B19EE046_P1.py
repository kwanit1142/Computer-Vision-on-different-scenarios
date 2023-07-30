import cv2
import numpy as np
import os
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from keras.datasets import cifar10
from tqdm import tqdm
from sklearn.metrics import precision_recall_curve, average_precision_score, classification_report
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def plot(K, eigenvectors):
  fig, ax = plt.subplots(1, K, figsize=(20, 5))
  for i in range(K):
    eigenface = eigenvectors[i].reshape((100, 100))
    ax[i].imshow(eigenface, cmap='gray')
    ax[i].set_title(f'Eigenface {i+1}')
    ax[i].axis('off')

def test(directory, pca_item, train_features):
  unknown_image = cv2.resize(cv2.imread(directory, 0),(100,100))
  unknown_features = pca_item.transform(np.array([unknown_image.flatten()]))
  distances = np.linalg.norm(train_features - unknown_features, axis=1)
  identity = np.argmin(distances) + 1
  print("The unknown image is most likely to be person No.", identity)

path = '/content/drive/MyDrive/CV_Assignment_3/face-lfw-train'
for img_name in os.listdir(path):
  img = cv2.imread(os.path.join(path, img_name))
  print(img.shape)
  cv2.imshow(str(img_name),img)

imgs_array = []
for img_name in os.listdir(path):
  img = cv2.imread(os.path.join(path, img_name),0)
  img = cv2.resize(img,(100,100))
  print(img.shape)
  cv2.imshow(str(img_name),img)
  imgs_array.append(img)

imgs_array = np.stack(imgs_array, axis=0).reshape(11,-1)
print(imgs_array.shape)

pca = PCA(n_components=11)
pca.fit(imgs_array)
train_features = pca.transform(imgs_array)
eigenvectors = pca.components_
print(eigenvectors.shape)

K=2
plot(K, eigenvectors)
K=3
plot(K, eigenvectors)
K=4
plot(K, eigenvectors)
K=5
plot(K, eigenvectors)
K=6
plot(K, eigenvectors)
K=7
plot(K, eigenvectors)
K=8
plot(K, eigenvectors)
K=9
plot(K, eigenvectors)
K=10
plot(K, eigenvectors)
K=11
plot(K, eigenvectors)

test('/content/drive/MyDrive/CV_Assignment_3/face-test/Copy of Atal_Bihari_Vajpayee_0019.jpg', pca, train_features)
test('/content/drive/MyDrive/CV_Assignment_3/face-test/Copy of Edward_Belvin_0001.jpg', pca, train_features)
test('/content/drive/MyDrive/CV_Assignment_3/face-test/Copy of shubman-gill.jpg', pca, train_features)
test('/content/drive/MyDrive/CV_Assignment_3/face-test/Copy of Priyanka_Chopra_0001.jpg', pca, train_features)
test('/content/drive/MyDrive/CV_Assignment_3/face-test/virat-kohli.jpg', pca, train_features)