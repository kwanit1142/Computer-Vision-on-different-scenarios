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

def histogram_plot(img, channel, title, bin_size, mode='rgb'):
  if mode=='rgb':
    img = img[:,:,channel]
  min_value = np.min(img)
  max_value = np.max(img)
  plt.hist(img.ravel(),int((max_value-min_value)/bin_size),[0,256])
  plt.title(title)
  plt.show()

dir = '/content/Kolabafort.jpg'

img = cv2.imread(dir)
gray_img = cv2.imread(dir,0)
print("Original Image:",img.shape)
cv2.imshow("Img",img)

histogram_plot(img, 0, 'Red-Channel Histograms Plot', 10)
histogram_plot(img, 1, 'Green-Channel Histograms Plot', 10)
histogram_plot(img, 2, 'Blue-Channel Histograms Plot', 10)
histogram_plot(gray_img, 0, 'Gray-Scale Histograms Plot', 10, 'gray')

global_img = cv2.equalizeHist(gray_img)
clahe_obj = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
clahe_img = clahe_obj.apply(gray_img)
comp_img = np.hstack((gray_img, global_img, clahe_img))
cv2.imshow("Comparison",comp_img)

histogram_plot(gray_img, 0, 'Original GrayScaled Image', 10, 'gray')
histogram_plot(global_img, 0, 'Global Histogram Equalized Image', 10, 'gray')
histogram_plot(clahe_img, 0, 'Adaptive Histogram Equalized Image', 10, 'gray')  