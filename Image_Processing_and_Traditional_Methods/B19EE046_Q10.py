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

dir1 = '/content/Assignment-1/Problem-10/Screenshot 2023-01-08 at 5.14.24 PM.png'
dir2 = '/content/Assignment-1/Problem-10/Screenshot 2023-01-08 at 5.14.47 PM.png'

img = cv2.imread(dir1)
print("Original Image:",img.shape)
cv2.imshow("Img1",img)

img1 = cv2.imread(dir2)
print("Original Image1:",img1.shape)
cv2.imshow("Img2",img1)

extracted_img = pytesseract.image_to_string(img)
print(extracted_img[-5:-2])

extracted_img1 = pytesseract.image_to_string(img1)
print(extracted_img1[-5:-2])