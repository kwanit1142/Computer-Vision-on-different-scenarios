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

def prompt(list_states, state1, state2):
  states=[]
  for i in range(0,len(list_states)):
    (_,Name,_) = list_states[i]
    if Name==state1 or Name==state2:
      states.append(list_states[i])
  return states[0], states[1]

def dist_calc(tuple1,tuple2):
  (coords1,Name1,conf1) = tuple1
  (coords2,Name2,conf2) = tuple2
  x_cord1 = int((coords1[0][0]+coords1[1][0])/2)
  x_cord2 = int((coords2[0][0]+coords2[1][0])/2)
  y_cord1 = int((coords1[0][1]+coords1[2][1])/2)
  y_cord2 = int((coords2[0][1]+coords2[2][1])/2)
  print("Distance between "+Name1+" and "+Name2+" = ",int((((x_cord1-x_cord2)**2)+((y_cord1-y_cord2)**2))**0.5))

dir = '/content/Assignment-1/Problem-2/india-map.jpg'
img = cv2.imread(dir)
print("Original Image:",img.shape)
cv2.imshow("Map",img)

kernel = np.array([[0, -1, 0],
                   [-1, 5,-1],
                   [0, -1, 0]])
imgc = cv2.filter2D(src=img, ddepth=-1, kernel=kernel)

reader = easyocr.Reader(['en'])
result = reader.readtext(imgc)
print(result)

t1, t2 = prompt(result, "KARNATAKA","TELANGANA")
dist_calc(t1, t2)