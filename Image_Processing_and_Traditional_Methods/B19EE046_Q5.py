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

def resize_gray(img, shape_tuple):
  resized_img = cv2.resize(img, shape_tuple)
  gray_img = cv2.cvtColor(resized_img, cv2.COLOR_RGB2GRAY)
  return gray_img

def salt_noise(img, prob):
  for i in range(img.shape[0]):
    for j in range(img.shape[1]):
      pixel_prob = random.random()
      if pixel_prob<=prob:
        img[i][j]=255
  return img

dir1 = '/content/Kolabafort.jpg'
dir2 = '/content/Konar_Dam.jpg'
dir3 = '/content/Kothandaramar_Temple.jpg'

img = cv2.imread()
print("Original Image:",img.shape)
cv2_imshow("Img",img)

img1 = cv2.imread('/content/Konar_Dam.jpg')
print("Original Image:",img1.shape)
cv2.imshow("Img1",img1)

img2 = cv2.imread('/content/Kothandaramar_Temple.jpg')
print("Original Image:",img2.shape)
cv2_imshow("Img2",img2)

print('Average of Pixels for Image-1:-', np.mean(img))
print('Average of Pixels for Image-2:-', np.mean(img1))
print('Average of Pixels for Image-3:-', np.mean(img2))

resize_img = cv2.resize(img, (256,256))
resize_img1 = cv2.resize(img1, (256,256))
resize_img2 = cv2.resize(img2, (256,256))

print('Average of Pixels for Image-1 after resizing:-', np.mean(resize_img))
print('Average of Pixels for Image-2 after resizing:-', np.mean(resize_img1))
print('Average of Pixels for Image-3 after resizing:-', np.mean(resize_img2))

avg_resize_img = (resize_img+resize_img1+resize_img2)/3
print('Average of Pixels for Images:-', np.mean(avg_resize_img))
cv2.imshow("Avg Image", avg_resize_img)

gr_img = resize_gray(img, (256,256))
gr_img1 = resize_gray(img1, (256,256))
gr_img2 = resize_gray(img2, (256,256))
comp = np.hstack((gr_img,gr_img1,gr_img2))
cv2.imshow("Resized_Gray Images",comp)

print('Average of Pixels for Image-1 after grayscaled resizing:-', np.mean(gr_img))
print('Average of Pixels for Image-2 after grayscaled resizing:-', np.mean(gr_img1))
print('Average of Pixels for Image-3 after grayscaled resizing:-', np.mean(gr_img2))

gr_avg_img = (gr_img+gr_img1+gr_img2)/3
print('Average of Pixels for Images after grayscaled resizing:-', np.mean(gr_avg_img))
cv2.imshow("Gray Image", gr_avg_img)

dif1 = gr_img1-gr_img2
dif2 = gr_img2-gr_img1
abso = np.abs(dif1)
comp = np.hstack((dif1,dif2,abso))
cv2.imshow("Difference Images",comp)

salt_img = salt_noise(img, 0.05)
salt_img1 = salt_noise(img1, 0.05)
salt_img2 = salt_noise(img2, 0.05)
salt_comp = np.hstack((salt_img,salt_img1,salt_img2))
cv2_imshow(salt_comp)

salt_img = salt_noise(gr_img, 0.05)
salt_img1 = salt_noise(gr_img1, 0.05)
salt_img2 = salt_noise(gr_img2, 0.05)
salt_comp = np.hstack((salt_img,salt_img1,salt_img2))
cv2.imshow("Noisy images",salt_comp)

denoised_img = cv2.medianBlur(salt_img, 3)
denoised_img1 = cv2.medianBlur(salt_img1, 3)
denoised_img2 = cv2.medianBlur(salt_img2, 3)
denoised_comp = np.hstack((denoised_img,denoised_img1,denoised_img2))
cv2.imshow("Denoised",denoised_comp)

kernel = np.array([[-1,-1,-1],[0,0,0],[1,1,1]])
conv_img = cv2.filter2D(img,-1,kernel)
conv_img1 = cv2.filter2D(img1,-1,kernel)
conv_img2 = cv2.filter2D(img2,-1,kernel)
conv_comp = np.hstack((conv_img,conv_img1,conv_img2))
cv2.imshow("Kernelized Images", conv_comp)

conv_denoised_img = cv2.filter2D(denoised_img,-1,kernel)
conv_denoised_img1 = cv2.filter2D(denoised_img1,-1,kernel)
conv_denoised_img2 = cv2.filter2D(denoised_img2,-1,kernel)
conv_denoised_comp = np.hstack((conv_denoised_img,conv_denoised_img1,conv_denoised_img2))
cv2.imshow("Kernelized Denoised Images",conv_denoised_comp)