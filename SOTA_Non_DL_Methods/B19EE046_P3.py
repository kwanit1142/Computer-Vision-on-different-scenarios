import cv2
import numpy as np
import os
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from keras.datasets import cifar10
from tqdm import tqdm
from sklearn.metrics import precision_recall_curve, average_precision_score, classification_report
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def fast_brightness(input_image, brightness):
    img = input_image.copy()
    cv2.convertScaleAbs(img, img, 1, brightness)
    return img

def sliding_window_detection(image, height, width, model_variable):
  img_h = image.shape[0]
  img_w = image.shape[1]
  color = (255,0,0)
  thickness = 1
  for height_index in tqdm(range(0,img_h-height,10)):
    for width_index in range(0,img_w-width,10):
      patch = cv2.resize(image[height_index:height_index+height, width_index:width_index+width, :],(128,128))
      feature = hog(patch, channel_axis=-1, block_norm='L1')
      prediction = model_variable.predict(feature.reshape(1,feature.shape[0]))
      if prediction==1:
        start_pt = (height_index, width_index)
        end_pt = (height_index+height, width_index+width)
        image = cv2.rectangle(image, start_pt, end_pt, color, thickness)
  cv2.imshow("Resultant Bounding Boxes",image)

def random_patches(img, patch, n, directory, name):
  img_height = img.shape[0]
  img_width = img.shape[1]
  height = patch.shape[0]
  width = patch.shape[1]
  print("Image Shape = ",img.shape)
  print("Patch Shape = ",patch.shape)
  height_perms = np.random.randint(low=0,high=img_height-height,size=n)
  width_perms = np.random.randint(low=0,high=img_width-width,size=n)
  for index in tqdm(range(n)):
    output = img[height_perms[index]:height_perms[index]+height,width_perms[index]:width_perms[index]+width,:]
    cv2.imwrite(os.path.join(directory,name)+'_'+str(index+1)+'.jpg',output)

def augmentations(patch, directory, name):
  cv2.imwrite(os.path.join(directory,name)+'_2.jpg',cv2.flip(patch,1))
  cv2.imwrite(os.path.join(directory,name)+'_3.jpg',cv2.GaussianBlur(patch,(3,3),0))
  cv2.imwrite(os.path.join(directory,name)+'_4.jpg',cv2.GaussianBlur(cv2.flip(patch,1),(3,3),0))
  cv2.imwrite(os.path.join(directory,name)+'_5.jpg',fast_brightness(patch,50))
  cv2.imwrite(os.path.join(directory,name)+'_6.jpg',fast_brightness(cv2.flip(patch,1),50))
  cv2.imwrite(os.path.join(directory,name)+'_7.jpg',fast_brightness(cv2.GaussianBlur(patch,(3,3),0),50))
  cv2.imwrite(os.path.join(directory,name)+'_8.jpg',fast_brightness(cv2.GaussianBlur(cv2.flip(patch,1),(3,3),0),50))

path = '/content/drive/MyDrive/CV_Assignment_3/Deer_Train_Patches'
for img_name in os.listdir(path):
  img = cv2.imread(os.path.join(path, img_name))
  print(img.shape)
  cv2.imshow("Images",img)

for img_name in tqdm(os.listdir(path)):
  img = cv2.imread(os.path.join(path, img_name))
  shape = img.shape
  augmentations(img, path, str(img_name[:len(img_name)-4]))

random_patches(cv2.imread('/content/drive/MyDrive/CV_Assignment_3/deer-train/deer 1.jpg'),
               cv2.imread('/content/drive/MyDrive/CV_Assignment_3/Deer_Train_Patches/deer 1.jpg'),
               22, '/content/drive/MyDrive/CV_Assignment_3/No_Deer_Train_Patches',
               'deer 1')

random_patches(cv2.imread('/content/drive/MyDrive/CV_Assignment_3/deer-train/deer 2.jpg'),
               cv2.imread('/content/drive/MyDrive/CV_Assignment_3/Deer_Train_Patches/deer 2.jpg'),
               8, '/content/drive/MyDrive/CV_Assignment_3/No_Deer_Train_Patches',
               'deer 2')

random_patches(cv2.imread('/content/drive/MyDrive/CV_Assignment_3/deer-train/deer 3.jpg'),
               cv2.imread('/content/drive/MyDrive/CV_Assignment_3/Deer_Train_Patches/deer 3.jpg'),
               8, '/content/drive/MyDrive/CV_Assignment_3/No_Deer_Train_Patches',
               'deer 3')

random_patches(cv2.imread('/content/drive/MyDrive/CV_Assignment_3/deer-train/deer 4.jpg'),
               cv2.imread('/content/drive/MyDrive/CV_Assignment_3/Deer_Train_Patches/deer 4_0.jpg'),
               8, '/content/drive/MyDrive/CV_Assignment_3/No_Deer_Train_Patches',
               'deer 4')

random_patches(cv2.imread('/content/drive/MyDrive/CV_Assignment_3/deer-train/deer 5.jpg'),
               cv2.imread('/content/drive/MyDrive/CV_Assignment_3/Deer_Train_Patches/deer 5_0.jpg'),
               8, '/content/drive/MyDrive/CV_Assignment_3/No_Deer_Train_Patches',
               'deer 5')

random_patches(cv2.imread('/content/drive/MyDrive/CV_Assignment_3/deer-train/deer 6.jpg'),
               cv2.imread('/content/drive/MyDrive/CV_Assignment_3/Deer_Train_Patches/deer 6_0.jpg'),
               22, '/content/drive/MyDrive/CV_Assignment_3/No_Deer_Train_Patches',
               'deer 6')

random_patches(cv2.imread('/content/drive/MyDrive/CV_Assignment_3/deer-train/deer 7.jpg'),
               cv2.imread('/content/drive/MyDrive/CV_Assignment_3/Deer_Train_Patches/deer 7.jpg'),
               8, '/content/drive/MyDrive/CV_Assignment_3/No_Deer_Train_Patches',
               'deer 7')

random_patches(cv2.imread('/content/drive/MyDrive/CV_Assignment_3/deer-train/deer 8.jpg'),
               cv2.imread('/content/drive/MyDrive/CV_Assignment_3/Deer_Train_Patches/deer 8.jpg'),
               8, '/content/drive/MyDrive/CV_Assignment_3/No_Deer_Train_Patches',
               'deer 8')

random_patches(cv2.imread('/content/drive/MyDrive/CV_Assignment_3/deer-train/deer 9.jpg'),
               cv2.imread('/content/drive/MyDrive/CV_Assignment_3/Deer_Train_Patches/deer 9_0.jpg'),
               22, '/content/drive/MyDrive/CV_Assignment_3/No_Deer_Train_Patches',
               'deer 9')

random_patches(cv2.imread('/content/drive/MyDrive/CV_Assignment_3/deer-train/deer 10.jpg'),
               cv2.imread('/content/drive/MyDrive/CV_Assignment_3/Deer_Train_Patches/deer 10_0.jpg'),
               22, '/content/drive/MyDrive/CV_Assignment_3/No_Deer_Train_Patches',
               'deer 10')

img_mat = []
lab_mat = []
for img in os.listdir('/content/drive/MyDrive/CV_Assignment_3/Deer_Train_Patches'):
  img_input = cv2.resize(cv2.imread(os.path.join('/content/drive/MyDrive/CV_Assignment_3/Deer_Train_Patches',img)),(128,128))
  x = hog(img_input,channel_axis=-1, block_norm='L1')
  img_mat.append(x)
  lab_mat.append(1)

for imgn in os.listdir('/content/drive/MyDrive/CV_Assignment_3/No_Deer_Train_Patches'):
  img_input = cv2.resize(cv2.imread(os.path.join('/content/drive/MyDrive/CV_Assignment_3/No_Deer_Train_Patches',imgn)),(128,128))
  x = hog(img_input,channel_axis=-1, block_norm='L1')
  img_mat.append(x)
  lab_mat.append(0)

img_mat = np.array(img_mat)
lab_mat = np.array(lab_mat)

x_train, x_test, y_train, y_test = train_test_split(img_mat, lab_mat, train_size=0.8,shuffle=True)

linear = SVC(kernel='linear')
linear.fit(x_train, y_train)
y_pred = linear.predict(x_test)
print(classification_report(y_pred, y_test))

poly = SVC(kernel='poly')
poly.fit(x_train, y_train)
y_pred = poly.predict(x_test)
print(classification_report(y_pred, y_test))

rbf = SVC(kernel='rbf')
rbf.fit(x_train, y_train)
y_pred = rbf.predict(x_test)
print(classification_report(y_pred, y_test))

linear = SVC(kernel='linear')
poly = SVC(kernel='poly')
rbf = SVC(kernel='rbf')
linear.fit(img_mat, lab_mat)
poly.fit(img_mat, lab_mat)
rbf.fit(img_mat, lab_mat)

### 1st Image (60% Window)

img = cv2.imread('/content/drive/MyDrive/CV_Assignment_3/deer-test/1.jpg')
height, width = int(0.6*img.shape[0]), int(0.6*img.shape[1])
sliding_window_detection(img, height, width, linear)

sliding_window_detection(img, height, width, poly)

sliding_window_detection(img, height, width, rbf)

### 1st Image (40% Window)

img = cv2.imread('/content/drive/MyDrive/CV_Assignment_3/deer-test/1.jpg')
height, width = int(0.4*img.shape[0]), int(0.4*img.shape[1])
sliding_window_detection(img, height, width, linear)

sliding_window_detection(img, height, width, poly)

sliding_window_detection(img, height, width, rbf)

### 1st Image (33% Window)

img = cv2.imread('/content/drive/MyDrive/CV_Assignment_3/deer-test/1.jpg')
height, width = int(img.shape[0]/3), int(img.shape[1]/3)
sliding_window_detection(img, height, width, linear)

sliding_window_detection(img, height, width, poly)

sliding_window_detection(img, height, width, rbf)

### 2nd Image (80% window)

img = cv2.imread('/content/drive/MyDrive/CV_Assignment_3/deer-test/2.jpg')
height, width = int(0.8*img.shape[0]), int(0.8*img.shape[1])
sliding_window_detection(img, height, width, linear)

sliding_window_detection(img, height, width, poly)

sliding_window_detection(img, height, width, rbf)

### 2nd Image (70% window)

img = cv2.imread('/content/drive/MyDrive/CV_Assignment_3/deer-test/2.jpg')
height, width = int(0.7*img.shape[0]), int(0.7*img.shape[1])
sliding_window_detection(img, height, width, linear)

sliding_window_detection(img, height, width, poly)

sliding_window_detection(img, height, width, rbf)

### 2nd Image (60% window)

img = cv2.imread('/content/drive/MyDrive/CV_Assignment_3/deer-test/2.jpg')
height, width = int(0.6*img.shape[0]), int(0.6*img.shape[1])
sliding_window_detection(img, height, width, linear)

sliding_window_detection(img, height, width, poly)

sliding_window_detection(img, height, width, rbf)

### 3rd Image

img = cv2.imread('/content/drive/MyDrive/CV_Assignment_3/deer-test/3.png')
height, width = int(0.6*img.shape[0]), int(0.6*img.shape[1])
sliding_window_detection(img, height, width, linear)

sliding_window_detection(img, height, width, poly)

sliding_window_detection(img, height, width, rbf)

### 4th Image (60% Window)

img = cv2.imread('/content/drive/MyDrive/CV_Assignment_3/deer-test/4.JPG')
height, width = int(0.6*img.shape[0]), int(0.6*img.shape[1])
sliding_window_detection(img, height, width, linear)

sliding_window_detection(img, height, width, poly)

sliding_window_detection(img, height, width, rbf)

### 4th Image (50% Window)

img = cv2.imread('/content/drive/MyDrive/CV_Assignment_3/deer-test/4.JPG')
height, width = int(0.5*img.shape[0]), int(0.5*img.shape[1])
sliding_window_detection(img, height, width, linear)

sliding_window_detection(img, height, width, poly)

sliding_window_detection(img, height, width, rbf)

### 4th Image (40% Window)

img = cv2.imread('/content/drive/MyDrive/CV_Assignment_3/deer-test/4.JPG')
height, width = int(0.4*img.shape[0]), int(0.4*img.shape[1])
sliding_window_detection(img, height, width, linear)

sliding_window_detection(img, height, width, poly)

sliding_window_detection(img, height, width, rbf)

### 5th Image (60% Window)

img = cv2.imread('/content/drive/MyDrive/CV_Assignment_3/deer-test/5.jpg')
height, width = int(0.6*img.shape[0]), int(0.6*img.shape[1])
sliding_window_detection(img, height, width, linear)

sliding_window_detection(img, height, width, poly)

sliding_window_detection(img, height, width, rbf)

### 5th Image (25% Window)

img = cv2.imread('/content/drive/MyDrive/CV_Assignment_3/deer-test/5.jpg')
height, width = int(0.25*img.shape[0]), int(0.25*img.shape[1])
sliding_window_detection(img, height, width, linear)

sliding_window_detection(img, height, width, poly)

sliding_window_detection(img, height, width, rbf)

### 5th Image (20% Window)

img = cv2.imread('/content/drive/MyDrive/CV_Assignment_3/deer-test/5.jpg')
height, width = int(0.2*img.shape[0]), int(0.2*img.shape[1])
sliding_window_detection(img, height, width, linear)

sliding_window_detection(img, height, width, poly)

sliding_window_detection(img, height, width, rbf)

### 6th Image (60% Window)

img = cv2.imread('/content/drive/MyDrive/CV_Assignment_3/deer-test/6.jpg')
height, width = int(0.6*img.shape[0]), int(0.6*img.shape[1])
sliding_window_detection(img, height, width, linear)

sliding_window_detection(img, height, width, poly)

sliding_window_detection(img, height, width, rbf)

### 6th Image (33% Window)

img = cv2.imread('/content/drive/MyDrive/CV_Assignment_3/deer-test/6.jpg')
height, width = int(img.shape[0]/3), int(img.shape[1]/3)
sliding_window_detection(img, height, width, linear)

sliding_window_detection(img, height, width, poly)

sliding_window_detection(img, height, width, rbf)

### 7th Image (60% Window)

img = cv2.imread('/content/drive/MyDrive/CV_Assignment_3/deer-test/7.JPG')
height, width = int(0.6*img.shape[0]), int(0.6*img.shape[1])
sliding_window_detection(img, height, width, linear)

sliding_window_detection(img, height, width, poly)

sliding_window_detection(img, height, width, rbf)

### 7th Image (33% Window)

img = cv2.imread('/content/drive/MyDrive/CV_Assignment_3/deer-test/7.JPG')
height, width = int(img.shape[0]/3), int(img.shape[1]/3)
sliding_window_detection(img, height, width, linear)

sliding_window_detection(img, height, width, poly)

sliding_window_detection(img, height, width, rbf)

### 8th Image (60% Window)

img = cv2.imread('/content/drive/MyDrive/CV_Assignment_3/deer-test/8.jpg')
height, width = int(0.6*img.shape[0]), int(0.6*img.shape[1])
sliding_window_detection(img, height, width, linear)

sliding_window_detection(img, height, width, poly)

sliding_window_detection(img, height, width, rbf)

### 8th Image (50% Window)

img = cv2.imread('/content/drive/MyDrive/CV_Assignment_3/deer-test/8.jpg')
height, width = int(0.5*img.shape[0]), int(0.5*img.shape[1])
sliding_window_detection(img, height, width, linear)

sliding_window_detection(img, height, width, poly)

sliding_window_detection(img, height, width, rbf)

### 8th Image (40% Window)

img = cv2.imread('/content/drive/MyDrive/CV_Assignment_3/deer-test/8.jpg')
height, width = int(0.4*img.shape[0]), int(0.4*img.shape[1])
sliding_window_detection(img, height, width, linear)

sliding_window_detection(img, height, width, poly)

sliding_window_detection(img, height, width, rbf)