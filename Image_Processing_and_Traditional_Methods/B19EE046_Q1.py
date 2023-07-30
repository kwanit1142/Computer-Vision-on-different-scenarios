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

def mask(img):
  img2 = np.zeros(img.shape)
  for i in range(img.shape[0]):
    for j in range(img.shape[1]):
      if img[i][j].all()!=0.0:
        img[i][j]=np.array([255.0,255.0,255.0])
        img2[i][j]=np.array([1.0,1.0,1.0])
  return img, img2

dir1 = '/content/Assignment-1/Problem-1/Spot_the_difference.png'

img = cv2.imread(dir1)
print("Original Image:",img.shape)
cv2.imshow("Img",img)

img1 = img[:img.shape[0],:int(img.shape[1]/2),:]
print("Left-Half Image:",img1.shape)
cv2.imshow("Img1",img)

img2 = img[:img.shape[0],int(img.shape[1]/2):,:]
print("Right-Half Image:",img2.shape)
cv2.imshow("Img2",img)

sub_img = np.abs(img1 - img2)
seg_mask, internal_mask = mask(sub_img)
cv2_imshow("Differences-Spotted", np.hstack((img1*internal_mask,img2*internal_mask)))
cv2_imshow("Segmented_Image:",img)