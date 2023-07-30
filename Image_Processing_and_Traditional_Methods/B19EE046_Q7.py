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

def sliding(img):
  list_horizontal = []
  reference_value=0
  for index in range(img.shape[1]):
    pixel_sum = np.apply_over_axes(np.sum, img[:,index,:], [0,1])
    list_horizontal.append(np.round(np.log(pixel_sum)).item(0))
    if index==0:
      reference_value = np.round(np.log(pixel_sum)).item(0)
  return reference_value, list_horizontal

def decision(reference_value, list_pixels):
  unique_list = np.unique(np.array(list_pixels)).tolist()
  count_less=0
  count_more=0
  sign=0
  for i in unique_list:
    if i!=reference_value and i<reference_value:
      count_less+=list_pixels.count(i)
      sign-=1
    if i!=reference_value and i>reference_value:
      count_more+=list_pixels.count(i)
      sign+=1
  if sign<0 and count_less>list_pixels.count(reference_value):
    print("Dark Text on Bright Background")
  elif sign>0 and count_more>list_pixels.count(reference_value):
    print("Bright Text on Dark Background")
  else:
    if count_less<count_more:
      print("Bright Text on Dark Background")
    else:
      print("Dark Text on Bright Background")

dir1 = '/content/Assignment-1/Problem-7/11_1.png'
dir2 = '/content/Assignment-1/Problem-7/27_2.png'

img = cv2.imread(dir1)
print("Original Image:",img.shape)
cv2.imshow("Img1",img)

img1 = cv2.imread(dir2)
print("Original Image:",img1.shape)
cv2.imshow("Img2",img)

ref, result = sliding(img)
print("Horizontally projected Value for background = ",ref)
plt.hist(result)
decision(ref, result)

ref1, result1 = sliding(img1)
print("Horizontally projected Value for background = ",ref1)
plt.hist(result1)
decision(ref1, result1)