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

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
def feature_detector(feature_mode='sift'):
  if feature_mode=='sift':
    fd = cv2.xfeatures2d.SIFT_create()
  elif feature_mode=='orb':
    fd = cv2.ORB_create()
  elif feature_mode=='akaze':
    fd = cv2.AKAZE_create()
  else:
    print("Its not available")
    return 0
  return fd

def K_Means_Clustering(num_clusters, descriptors):
  Vocabulary_Space = KMeans(num_clusters, n_init=10)
  Vocabulary_Space.fit(descriptors)
  return Vocabulary_Space

def textons(data_train, feature_mode='sift', num_features = 50):
  fd = feature_detector(feature_mode)
  dictionary = []
  for image_index in tqdm(range(data_train.shape[0])):
    des = fd.detectAndCompute(data_train[image_index],None)[1]
    if des is not None and des.shape[0] > num_features:
      des = des[:num_features]
    elif des is None:
      if feature_mode=='sift':
        des = np.zeros((num_features, 128))
      if feature_mode=='orb':
        des = np.zeros((num_features, 32))
      if feature_mode=='akaze':
        des = np.zeros((num_features, 61))
    else:
      if feature_mode=='sift':
        des = np.vstack((des, np.zeros((num_features - des.shape[0], 128))))
      if feature_mode=='orb':
        des = np.vstack((des, np.zeros((num_features - des.shape[0], 32))))
      if feature_mode=='akaze':
        des = np.vstack((des, np.zeros((num_features - des.shape[0], 61))))
    dictionary.append(des)
  textons_dictionary = np.vstack(dictionary)
  return textons_dictionary

def Visual_BoW(data_train, vocabulary_space, feature_mode='sift', num_features = 50, num_clusters = 100):
  fd = feature_detector(feature_mode)
  BoW = []
  for image_index in tqdm(range(data_train.shape[0])):
    des = fd.detectAndCompute(data_train[image_index],None)[1]
    if des is not None and des.shape[0] > num_features:
      des = des[:num_features]
    elif des is None:
      if feature_mode=='sift':
        des = np.zeros((num_features, 128))
      if feature_mode=='orb':
        des = np.zeros((num_features, 32))
      if feature_mode=='akaze':
        des = np.zeros((num_features, 61))
    else:
      if feature_mode=='sift':
        des = np.vstack((des, np.zeros((num_features - des.shape[0], 128))))
      if feature_mode=='orb':
        des = np.vstack((des, np.zeros((num_features - des.shape[0], 32))))
      if feature_mode=='akaze':
        des = np.vstack((des, np.zeros((num_features - des.shape[0], 61))))
    bow = np.bincount(vocabulary_space.predict(des.astype('float64')), minlength=num_clusters)
    BoW.append(bow)
  BoW_representation = np.vstack(BoW)
  return BoW_representation

def K_Neighbors(num_outputs, BoW_train):
  Output_Space = NearestNeighbors(n_neighbors=num_outputs)
  Output_Space.fit(BoW_train)
  return Output_Space

def image_search(query, data_train, vocabulary_space, output_space, feature_mode='sift', num_features = 50, num_clusters = 100):
  fd = feature_detector(feature_mode)
  feature = fd.detectAndCompute(query,None)[1]
  if feature is not None and feature.shape[0] > num_features:
    feature = feature[:num_features]
  elif feature is None:
    if feature_mode=='sift':
      feature = np.zeros((num_features, 128))
    if feature_mode=='orb':
      feature = np.zeros((num_features, 32))
    if feature_mode=='akaze':
      feature = np.zeros((num_features, 61))
  else:
    if feature_mode=='sift':
      feature = np.vstack((feature, np.zeros((num_features - feature.shape[0], 128))))
    if feature_mode=='orb':
      feature = np.vstack((feature, np.zeros((num_features - feature.shape[0], 32))))
    if feature_mode=='akaze':
      feature = np.vstack((feature, np.zeros((num_features - feature.shape[0], 61))))
  bow = np.bincount(vocabulary_space.predict(feature.astype('float64')), minlength=num_clusters).reshape(1, -1)
  _, indices = output_space.kneighbors(bow)
  similar_images = data_train[indices.flatten()]
  return similar_images, indices.flatten()

def Search_Engine(x_train, x_test, y_train, y_test, feature_mode='sift', num_features = 50, num_clusters=100, num_outputs=5, test_mode='individual'):
  dictionary = textons(x_train, feature_mode, num_features)
  print('Dictionary Done')
  vocabulary_space = K_Means_Clustering(num_clusters, dictionary)
  print('Vocabulary Done')
  BoW = Visual_BoW(x_train, vocabulary_space, feature_mode, num_features, num_clusters)
  print('Bag of Words Done')
  output_space = K_Neighbors(num_outputs, BoW)
  print('Output Space Done')
  if test_mode=='individual':
    #query_img = x_test[0]
    query_img = x_test[500]
    similar_images,_ = image_search(query_img, x_train, vocabulary_space, output_space, feature_mode, num_features, num_clusters)
    cv2.imshow("Query Image", cv2.resize(query_img,(100,100)))
    for resultant_image in similar_images:
      cv2.imshow("Similar Images",cv2.resize(resultant_image,(100,100)))
  elif test_mode=='result':
    test_bow = Visual_BoW(x_test, vocabulary_space, feature_mode, num_features, num_clusters)
    dist, ind = output_space.kneighbors(test_bow)
    num_test_samples = y_test.shape[0]
    y_scores = np.zeros((num_test_samples, num_outputs))
    y_true = np.zeros(num_test_samples)
    for i in tqdm(range(num_test_samples)):
        y_scores[i] = np.sum(y_train[ind[i]] == y_test[i], axis=0)
        y_true[i] = np.where(y_train == y_test[i])[0][0]
    for class_ind in range(10):
      precision, recall, thresholds = precision_recall_curve(y_true, y_scores[:, 0], pos_label=class_ind)
      print("\nFor Class No.", class_ind+1)
      plt.plot(recall, precision, lw=2)
      plt.xlabel('Recall')
      plt.ylabel('Precision')
      plt.ylim([0.0, 1.05])
      plt.xlim([0.0, 1.0])
      plt.title('Precision-Recall Curve')
      plt.legend(loc="lower left")
      plt.show()
  else:
    print("That mode is not available")

Search_Engine(x_train, x_test, y_train, y_test, 'orb', 5, 100, 5, 'individual')
Search_Engine(x_train, x_test, y_train, y_test, 'akaze', 5, 100, 5, 'individual')
Search_Engine(x_train, x_test, y_train, y_test, 'sift', 5, 100, 5, 'individual')
Search_Engine(x_train, x_test, y_train, y_test, 'orb', 5, 100, 5, 'result')
Search_Engine(x_train, x_test, y_train, y_test, 'akaze', 5, 100, 5, 'result')
Search_Engine(x_train, x_test, y_train, y_test, 'sift', 5, 100, 5, 'result')