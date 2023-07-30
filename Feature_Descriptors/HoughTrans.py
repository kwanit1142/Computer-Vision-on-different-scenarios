import cv2
import numpy as np
import os
import time
import math

def hough_transform(img):
    def loop_body(x, y):
        if edges[x][y] != 0:
            for i in range(len(theta_range)):
                r = x * np.cos(theta_range[i]) + y * np.sin(theta_range[i])
                accumulator[int(r + diagonal)][i] += 1
    
    edges = cv2.Canny(img, 50, 200, apertureSize=3)
    theta_range = np.deg2rad(np.arange(-90, 90))
    h, w = edges.shape
    diagonal = int(math.sqrt(h ** 2 + w ** 2))
    rho = np.linspace(-diagonal, diagonal, diagonal * 2)
    accumulator = np.zeros((len(rho), len(theta_range)))
    np.fromfunction(np.vectorize(loop_body), (h, w), dtype=int)
    threshold = 0.8 * np.max(accumulator)
    y_idxs, x_idxs = np.where(accumulator >= threshold)
    rhos = rho[y_idxs]
    thetas = theta_range[x_idxs]
    lines = np.column_stack((rhos, thetas))
    return lines

def draw_lines(img, lines):
    for line in lines:
        r, theta = line
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * r
        y0 = b * r
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    return img

def custom_hough_transform(img):
    t = time.time()
    lines = hough_transform(img)
    img = draw_lines(img, lines)
    new_t = time.time()
    return (new_t - t), img

def opencv_hough_transform(img):
    t = time.time()
    edges = cv2.Canny(img, 50, 200, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)
    for r_theta in lines:
        arr = np.array(r_theta[0], dtype=np.float64)
        r, theta = arr
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * r
        y0 = b * r
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    new_t = time.time()
    return (new_t - t), img

img = cv2.imread('/content/drive/MyDrive/CV_Assignment_2/Problem-1/logo matching/Ex2/logos/hp.jpg', 0)
custom_t, custom_img = custom_hough_transform(img)
opencv_t, opencv_img = opencv_hough_transform(img)
cv2.imshow("Custom_IMG",custom_img)
print(custom_t)
cv2.imshow("Opencv_IMG",opencv_img)
print(opencv_t)
cv2.imshow("IMG",custom_img-opencv_img)