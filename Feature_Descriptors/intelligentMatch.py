import cv2
import numpy as np
import os
import time
import math

img_ref = cv2.imread('/content/drive/MyDrive/CV_Assignment_2/Problem-3/example-1/reference.png')
img_p = cv2.resize(cv2.imread('/content/drive/MyDrive/CV_Assignment_2/Problem-3/example-1/perfect/penetration_checkLin.png'),(img_ref.shape[1],img_ref.shape[0]))
img_f = cv2.resize(cv2.imread('/content/drive/MyDrive/CV_Assignment_2/Problem-3/example-1/faulty/Element_Optimised_Colour_ShapeLin.png'),(img_ref.shape[1],img_ref.shape[0]))
img_refg = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)
img_pg = cv2.cvtColor(img_p, cv2.COLOR_BGR2GRAY)
img_fg = cv2.cvtColor(img_f, cv2.COLOR_BGR2GRAY)
cv2.imshow("IMG1",img_ref - img_p)
print(" ")
cv2.imshow("IMG2",img_ref - img_f)
print(" ")
cv2.imshow("g1IMG",img_refg - img_pg)
print(" ")
cv2.imshow("g2IMG",img_refg - img_fg)

cv2.imshow("IMG3",(img_ref/img_p)**2)
print(" ")
cv2.imshow("IMG4",(img_ref/img_f)**2)
print(" ")
cv2.imshow("g3IMG",(img_refg/img_pg)**5)
print(" ")
cv2.imshow("g4IMG",(img_refg/img_fg)**5)

cv2.imshow("IMG5",np.log(img_ref/img_p))
print(" ")
cv2.imshow("IMG6",np.log(img_ref/img_f))

img_ref1 = cv2.imread('/content/drive/MyDrive/CV_Assignment_2/Problem-3/example-2/Orginal.png')
img_p1 = cv2.resize(cv2.imread('/content/drive/MyDrive/CV_Assignment_2/Problem-3/example-2/perfect/Orginal_Lin.png'),(img_ref.shape[1],img_ref.shape[0]))
img_f1 = cv2.resize(cv2.imread('/content/drive/MyDrive/CV_Assignment_2/Problem-3/example-2/faulty/2ndOrderElements.png'),(img_ref.shape[1],img_ref.shape[0]))
img_refg1 = cv2.cvtColor(img_ref1, cv2.COLOR_BGR2GRAY)
img_pg1 = cv2.cvtColor(img_p1, cv2.COLOR_BGR2GRAY)
img_fg1 = cv2.cvtColor(img_f1, cv2.COLOR_BGR2GRAY)
cv2.imshow("IMG1",img_ref1 - img_p1)
print(" ")
cv2.imshow("IMG2",img_ref1 - img_f1)
print(" ")
cv2.imshow("g1IMG",img_refg1 - img_pg1)
print(" ")
cv2.imshow("g2IMG",img_refg1 - img_fg1)

cv2.imshow("IMG3",(img_ref1/img_p1)**5)
print(" ")
cv2.imshow("IMG4",(img_ref1/img_f1)**5)
print(" ")
cv2.imshow("g3IMG",(img_refg1/img_pg1)**2)
print(" ")
cv2.imshow("g4IMG",(img_refg1/img_fg1)**2)