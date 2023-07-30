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

def grad_matrix(img):
  im = np.float32(img)/ 255.0
  x = cv2.Sobel(im, cv2.CV_32F, 1, 0, ksize=1)
  y = cv2.Sobel(im, cv2.CV_32F, 0, 1, ksize=1)
  mag, angle = cv2.cartToPolar(x, y, angleInDegrees=True)
  return x, y, mag, angle

def line(img,gray_img):
  edges = cv2.Canny(gray_img,50,150,apertureSize=3)
  lines_list =[]
  mag_list=[]
  ang_list=[]
  mag=0
  lines = cv2.HoughLinesP(edges,1,np.pi/180,threshold=75,minLineLength=0,maxLineGap=10)
  for points in lines:
    x1,y1,x2,y2=points[0]
    cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)
    lines_list.append([(x1,y1),(x2,y2)])
    mag_list.append((((x1-x2)**2)+((y1-y2)**2))**0.5)
    ang_list.append((y2-y1)/(x2-x1))
  cv2_imshow(img)
  hand1 = max(mag_list)
  ang1 = ang_list[mag_list.index(hand1)]
  print(hand1,ang1)

dir1 = '/content/Assignment-1/Problem-4/clock_1.png'
dir1 = '/content/Assignment-1/Problem-4/clock_2.jpg'

img_rgb = cv2.imread(dir1)
img = cv2.imread(dir1,0)
print("Original Image1:",img.shape)
cv2.imshow("Img1",img)

img1_rgb = cv2.imread(dir2)
img1 = cv2.imread(dor2,0)
print("Original Image2:",img1.shape)
cv2.imshow("Img2",img)

gx,gy,mag,ang = grad_matrix(img)
plt.hist(ang.reshape(-1),bins=30)
plt.hist(gx.reshape(-1),bins=30)
plt.hist(gy.reshape(-1),bins=30)

gx,gy,mag,ang = grad_matrix(img1)
plt.hist(ang.reshape(-1),bins=30)
plt.hist(gx.reshape(-1),bins=30)
plt.hist(gy.reshape(-1),bins=30)

line(cv2.resize(img_rgb,(512,512)),cv2.resize(img,(512,512)))
line(img1_rgb,img1)