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

def peri_area(img):
  peri=0
  for i in range(img.shape[0]):
    for j in range(img.shape[1]):
      if img[i][j].all()==0:
        peri+=1
  radius = int((peri/(2*math.pi)))
  area = int(math.pi*radius**2)
  return area, peri, radius

def hough(img):
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  blurred = cv2.blur(gray, (3, 3))
  detected_circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, 20, param1 = 50, param2 = 30, minRadius = 0, maxRadius = img.shape[1])
  actual_circles = np.uint16(np.around(detected_circles))
  for pt in detected_circles[0, :]:
    x, y, radius = pt[0], pt[1], pt[2]
    break
  return int(radius), int(2*math.pi*radius), int(math.pi*(radius**2))

dir = '/content/circle-updated.png'

img = np.asarray(PIL.Image.open(dir))
print("Original Image:",img.shape)
cv2.imshow("Img",img[:,:,:3])

img_actual = img[:,:,:3]
area, peri, rad = peri_area(img_actual)
print("Area of Circle:- ",area)
print("Perimeter of Circle:- ",peri)
print("Radius of Circle:- ",rad)

img_actualb = cv2.blur(img[:,:,:3],(3,3))
cv2_imshow(img_actualb)
areab, perib, radb = peri_area(img_actualb)
print("Area of Circle:- ",areab)
print("Perimeter of Circle:- ",perib)
print("Radius of Circle:- ",radb)

radc, peric, areac = hough(img[:,:,:3])
print("Area of Circle:- ",areac)
print("Perimeter of Circle:- ",peric)
print("Radius of Circle:- ",radc)