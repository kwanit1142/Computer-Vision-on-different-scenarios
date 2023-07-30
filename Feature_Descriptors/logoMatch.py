import cv2
import numpy as np
import os
import time
import math

def feature_detector_pairwise(img1, img2, mode='sift'):
  img1 = cv2.resize(img1,(int(img1.shape[0]*0.75),int(img1.shape[1]*0.5)))
  img2 = cv2.resize(img2,(int(img2.shape[0]*0.75),int(img2.shape[1]*0.5)))
  g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
  g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
  if mode=='sift':
    fd = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = fd.detectAndCompute(g1, None)
    kp2, des2 = fd.detectAndCompute(g2, None)
    matches = feature_matcher(des1,des2,'BF_plain')
    matches = sorted(matches, key = lambda x:x.distance)
    img = cv2.drawMatches(img1, kp1, img2, kp2, matches[:15], img2, flags=2)
    cv2.imshow("IMG",img)
  elif mode=='kaze':
    fd = cv2.KAZE_create()
    kp1, des1 = fd.detectAndCompute(g1, None)
    kp2, des2 = fd.detectAndCompute(g2, None)
    matches = feature_matcher(des1,des2,'BF_plain')
    matches = sorted(matches, key = lambda x:x.distance)
    img = cv2.drawMatches(img1, kp1, img2, kp2, matches[:15], img2, flags=2)
    cv2.imshow("IMG",img)
  elif mode=='akaze':
    fd = cv2.AKAZE_create()
    kp1, des1 = fd.detectAndCompute(g1, None)
    kp2, des2 = fd.detectAndCompute(g2, None)
    matches = feature_matcher(des1,des2,'BF_plain')
    matches = sorted(matches, key = lambda x:x.distance)
    img = cv2.drawMatches(img1, kp1, img2, kp2, matches[:15], img2, flags=2)
    cv2.imshow("IMG",img)
  else:
    print("Its not available")

def feature_matcher(p1,p2,mode='BF'):
  if mode=='BF':
    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.knnMatch(p1, p2, k=2)
  elif mode=='BF_plain':
    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.match(p1, p2)
  elif mode=='FLANN':
    index_params = dict(algorithm=0, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(p1, p2, k=2)
  elif mode=='HTM':
    matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
    matcher.setCrossCheck(True)
    matches = matcher.match(p1, p2)
  else:
    print("Not Available")
    return None
  return matches

scene_img = cv2.imread('/content/drive/MyDrive/CV_Assignment_2/Problem-1/logo matching/Ex1/levis.jpg')
logos_list = os.listdir('/content/drive/MyDrive/CV_Assignment_2/Problem-1/logo matching/Ex1/logos')

for i in logos_list:
  logo_img = cv2.imread(os.path.join('/content/drive/MyDrive/CV_Assignment_2/Problem-1/logo matching/Ex1/logos',i))
  feature_detector_pairwise(scene_img,logo_img,'sift')

for iN in logos_list:
  logo_img = cv2.imread(os.path.join('/content/drive/MyDrive/CV_Assignment_2/Problem-1/logo matching/Ex1/logos',iN))
  feature_detector_pairwise(scene_img,logo_img,'kaze')

for iNN in logos_list:
  logo_img = cv2.imread(os.path.join('/content/drive/MyDrive/CV_Assignment_2/Problem-1/logo matching/Ex1/logos',iNN))
  feature_detector_pairwise(scene_img,logo_img,'akaze')

scene_img = cv2.imread('/content/drive/MyDrive/CV_Assignment_2/Problem-1/logo matching/Ex2/starbucks.jpeg')
logos_list = os.listdir('/content/drive/MyDrive/CV_Assignment_2/Problem-1/logo matching/Ex2/logos')

for i in logos_list:
  logo_img = cv2.imread(os.path.join('/content/drive/MyDrive/CV_Assignment_2/Problem-1/logo matching/Ex2/logos',i))
  feature_detector_pairwise(scene_img,logo_img,'sift')

for iN in logos_list:
  logo_img = cv2.imread(os.path.join('/content/drive/MyDrive/CV_Assignment_2/Problem-1/logo matching/Ex2/logos',iN))
  feature_detector_pairwise(scene_img,logo_img,'kaze')

for iNN in logos_list:
  logo_img = cv2.imread(os.path.join('/content/drive/MyDrive/CV_Assignment_2/Problem-1/logo matching/Ex2/logos',iNN))
  feature_detector_pairwise(scene_img,logo_img,'akaze')