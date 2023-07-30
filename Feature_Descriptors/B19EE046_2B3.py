import cv2
import numpy as np
import os
import time
import math

def feature_detector(g1,g2,g3,mode='sift'):
  if mode=='sift':
    fd = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = fd.detectAndCompute(g1, None)
    kp2, des2 = fd.detectAndCompute(g2, None)
    kp3, des3 = fd.detectAndCompute(g3, None)
  elif mode=='orb':
    fd = cv2.ORB_create()
    kp1, des1 = fd.detectAndCompute(g1, None)
    kp2, des2 = fd.detectAndCompute(g2, None)
    kp3, des3 = fd.detectAndCompute(g3, None)
  elif mode=='akaze':
    fd = cv2.AKAZE_create()
    kp1, des1 = fd.detectAndCompute(g1, None)
    kp2, des2 = fd.detectAndCompute(g2, None)
    kp3, des3 = fd.detectAndCompute(g3, None)
  else:
    print("Its not available")
    return 0,0,0,0,0,0
  return kp1,kp2,kp3,des1,des2,des3

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

def matcher_test(matches,mode='Ratio'):
  good_matches = []
  if mode=='Ratio':
    for m, n in matches:
      if m.distance < 0.75 * n.distance:
        good_matches.append(m)
  elif mode=='Lowe':
    for m in matches:
      if len(m) == 2 and m[0].distance < m[1].distance * 0.7:
        good_matches.append(m[0])
  else:
    print("Not available")
  return good_matches

# Load images
img1 = cv2.imread('/content/drive/MyDrive/CV_Assignment_2/problem-3/1a.jpeg')
img2 = cv2.imread('/content/drive/MyDrive/CV_Assignment_2/problem-3/1b.jpeg')
img3 = cv2.imread('/content/drive/MyDrive/CV_Assignment_2/problem-3/1c.jpeg')
cv2.imshow("IMG1",img1)
cv2.imshow("IMG2",img2)
cv2.imshow("IMG3",img3)

# Convert to grayscale
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
gray3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
cv2.imshow("gIMG1",gray1)
cv2.imshow("gIMG2",gray2)
cv2.imshow("gIMG3",gray3)

kp1,kp2,kp3,des1,des2,des3 = feature_detector(gray1,gray2,gray3,'orb')
kp_img1 = cv2.drawKeypoints(gray1, kp1, None, flags=0)
kp_img2 = cv2.drawKeypoints(gray2, kp2, None, flags=0)
kp_img3 = cv2.drawKeypoints(gray3, kp3, None, flags=0)
cv2.imshow("kIMG1",kp_img1)
cv2.imshow("kIMG2",kp_img2)
cv2.imshow("kIMG3",kp_img3)

matches = feature_matcher(des1,des2,'BF')
good_matches = matcher_test(matches,'Ratio')
src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

# determine output size based on input images and transformation matrix
h1, w1 = img1.shape[:2]
h2, w2 = img2.shape[:2]
pts1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
pts2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
pts2_transformed = cv2.perspectiveTransform(pts2, H)
pts = np.concatenate((pts1, pts2_transformed), axis=0)
[x_min, y_min] = np.int32(pts.min(axis=0).ravel() - 0.5)
[x_max, y_max] = np.int32(pts.max(axis=0).ravel() + 0.5)
warp_size = (x_max - x_min, y_max - y_min)

# adjust transformation matrix to shift image content to left
shift = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])
H_shifted = shift.dot(H)

# warp images and combine
warp_img = cv2.warpPerspective(img1, H_shifted, warp_size)
warp_img[-y_min:h2-y_min, -x_min:w2-x_min] = img2
cv2.imshow("wIMG",warp_img)