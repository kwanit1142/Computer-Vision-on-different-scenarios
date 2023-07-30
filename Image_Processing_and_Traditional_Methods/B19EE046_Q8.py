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

def detect(method, img_dir1, img_dir2, threshold, technique):
  img_ref = cv2.imread(img_dir1)
  img = cv2.imread(img_dir1,0)
  img1 = cv2.imread(img_dir2,0)
  w, h = img1.shape[1], img1.shape[0]
  res = cv2.matchTemplate(img, img1, technique)
  loc = np.where(res >= threshold)
  for pt in zip(*loc[::-1]):
    cv2.rectangle(img_ref, pt, (pt[0] + w, pt[1] + h), (255, 0, 0), 2)
  cv2.imshow(method, img_ref)

dir1 = '/content/Name.jpg'
dir2 = '/content/Letter.jpg'
dir3 = '/content/Letter_I.jpg'

img_ref = cv2.imread(dir1)
img = cv2.imread(dir1,0)
print("Name Image:",img.shape)
cv2.imshow("Img",img)

img1 = cv2.imread(dir2,0)
print("Letter_1 Image:",img1.shape)
cv2.imshow("Template1",img1)

img2 = cv2.imread(dir3,0)
print("Letter-I Image:",img2.shape)
cv2.imshow("Template2",img2)

methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR','cv2.TM_CCORR_NORMED']

detect(methods[1],dir1,dir2,0.8,cv2.TM_CCOEFF_NORMED)
detect(methods[1],dir1,dir2,0.7,cv2.TM_CCOEFF_NORMED)
detect(methods[1],dir1,dir2,0.6,cv2.TM_CCOEFF_NORMED)
detect(methods[1],dir1,dir2,0.5,cv2.TM_CCOEFF_NORMED)

detect(methods[1],dir1,dir3,0.8,cv2.TM_CCOEFF_NORMED)
detect(methods[1],dir1,dir3,0.7,cv2.TM_CCOEFF_NORMED)
detect(methods[1],dir1,dir3,0.6,cv2.TM_CCOEFF_NORMED)
detect(methods[1],dir1,dir3,0.5,cv2.TM_CCOEFF_NORMED)

detect(methods[0],dir1,dir2,0.999,cv2.TM_CCOEFF)
detect(methods[0],dir1,dir3,0.999,cv2.TM_CCOEFF)

detect(methods[2],dir1,dir2,0.999,cv2.TM_CCORR)
detect(methods[2],dir1,dir3,0.999,cv2.TM_CCORR)

detect(methods[3],dir1,dir2,0.97,cv2.TM_CCORR_NORMED)
detect(methods[3],dir1,dir2,0.9675,cv2.TM_CCORR_NORMED)
detect(methods[3],dir1,dir2,0.965,cv2.TM_CCORR_NORMED)
detect(methods[3],dir1,dir2,0.9625,cv2.TM_CCORR_NORMED)

detect(methods[3],dir1,dir3,0.995,cv2.TM_CCORR_NORMED)
detect(methods[3],dir1,dir3,0.99,cv2.TM_CCORR_NORMED)
detect(methods[3],dir1,dir3,0.985,cv2.TM_CCORR_NORMED)
detect(methods[3],dir1,dir3,0.98,cv2.TM_CCORR_NORMED)