# Computer Vision on different scenarios

These Notebooks with their Question Statement and Reports, came under the course CSL7360, taken by Prof. Anand Mishra.

## Lab-1 (Image Processing and Traditional Methods)

Question-1

![download](https://github.com/kwanit1142/Computer-Vision-on-different-scenarios/assets/54277039/26be8f03-e9d5-40b9-a84c-773791032a24)

Given a stitched image containing 2 very similar scenes, find out the differences.

a.) Submit your implementation

b.) Write down your algorithm in brief

c.) Show the image where differences are suitably marked

d.) Write down scenarios when your implementation may not work

Question-2

![download](https://github.com/kwanit1142/Computer-Vision-on-different-scenarios/assets/54277039/8bb153d6-f80b-4c8c-b165-17b9a1c7ec62)

Given an image of the map of india, find out the pixel distance between the 2 states [Hint: Use Off-the-Shell OCR].

a.) Submit your implementation

b.) Write down the limitations of your approach

Question-3

![circle_noun_001_02738](https://github.com/kwanit1142/Computer-Vision-on-different-scenarios/assets/54277039/203712b6-21db-4c79-a7cb-b22d2c25bd14)

Given an image of a circle, find out the area and perimeter in the pixel unit.

a.) Submit your implementation such that it takes the image file as an argument and prints the area and perimeter in new lines.

Question-4

![istockphoto-1175043035-612x612](https://github.com/kwanit1142/Computer-Vision-on-different-scenarios/assets/54277039/f7944f2f-d3c8-4298-b27a-ed786c2c36f9)

Given an image of a clock, find out the angle between the hour and minute hands.

a.) Submit your implementation

b.) Write down your approach to finding out the angle

c.) Write down the limitations of your approach.

Question-5

![Google_Landmarks_Dataset_v2-0000004608-31bcf8ba](https://github.com/kwanit1142/Computer-Vision-on-different-scenarios/assets/54277039/16d9d3d2-bcb2-4c5a-836d-fc242e1c6251)

Choose 3 images of a world landmark from the Google Landmark dataset. The name of your chosen landmark should begin with the 1st letter of your 1st name.

a.) Resize all images to 256 x 256 and Convert it to gray

b.) Show the average of all 3 images

c.) Substract Image-1 with Image-2

d.) Add Salt Noise with 5% probability in on of the images

e.) Remove the noise

f.) Use the following 3 x 3 kernel [-1,-1,-1,0,0,0,1,1,1] for performing convolution in one of the images and show the output.

Question-6

![0_Yf6jSy8y3QHHhAws](https://github.com/kwanit1142/Computer-Vision-on-different-scenarios/assets/54277039/b0162015-d2ac-482d-9c2c-fe1032d8070a)

You will be given 100 handwritten images of 0 and 1. You have to compute horizontal projection profile features and use Nearest Neighbour and SVM Classifiers to recognize the digits. Report accuracy and show some visual examples (Keep only images of 0s and 1s from MNIST)

Question-7

![less-light-reflect-text](https://github.com/kwanit1142/Computer-Vision-on-different-scenarios/assets/54277039/7b7f2728-7f58-4f69-b8c9-6f51cc5f9829)

Given a word image, find out if the word is bright text on a dark background or dark text on bright background.

Question-8

![template_ccoeffn_2](https://github.com/kwanit1142/Computer-Vision-on-different-scenarios/assets/54277039/81140a32-15e1-4e54-86ea-2d77ee5a9251)

Write your name in capital letters on a piece of white paper and a random letter from your name. Click photographs of these. Implement the Template Matching Algorithm and discuss your observation.

Question-9

![Example-of-histogram-equalization-a-The-input-image-and-b-its-graylevel-histogram](https://github.com/kwanit1142/Computer-Vision-on-different-scenarios/assets/54277039/88ba41f9-22ff-4680-9e04-dbad5f311fbd)

Choose one image from Problem-5. Show histogram of pixel values with bin size 10. Perform histogram equalization and show the output image.

Question-10

![download](https://github.com/kwanit1142/Computer-Vision-on-different-scenarios/assets/54277039/6f180922-78ee-40ed-90fe-31b3c39f0c53)

You will be given an image of a mobile number. Use Off-the-shell OCR and find out the last 3 digits of the mobile number.

## Lab-2 (Feature Descriptors)

Question-1

![1_UqpTAesCJHYJZJw9PpN2ZQ](https://github.com/kwanit1142/Computer-Vision-on-different-scenarios/assets/54277039/6c0460e1-a240-4459-9b4e-d3390f3da53f)

You are given a scene image containing a logo, and a gallery containing reference logos for 10 business brands. Find out which business brand is present in the scene. Try out 3 different approaches and compare them.

a.) Submit your implementation. The file name of your python file should be logoMatch.py and it should take the scene image, logo image, and approach name as input and either show the matched region or say not enough match points found.

b.) Show your results qualitatively in the report and write down your observation.

Question-2

![1_pSAwZyau08leeYmTM9203Q](https://github.com/kwanit1142/Computer-Vision-on-different-scenarios/assets/54277039/1bebd421-6662-42d8-bf22-e7062e8233e8)

Implement Hough Transform for line detection from scratch. Compare the result of openCV implementation v/s your implementation (both speed and performance-wise) on the picture of your choice.

a.) Submit your implementation. The file name of your python file should be HoughTrans.py.

b.) Show your results qualitatively in the report and write down your observation.

Question-3

![Defect-detection-on-same-Noisy-SEM-image](https://github.com/kwanit1142/Computer-Vision-on-different-scenarios/assets/54277039/fc3f1fc8-4a21-4ffa-8e37-b9e3b5941e59)

A manufacturing company in Bangalore, came up with the following Problem Statement. They have one reference desigm image for a part of equipment and a probe image of either faulty or perfect. You need to identify the faulty image and show the defective region.

a.) Submit your implementation. The file name of your python file should be intelligentMatch.py, it should take 2 images (reference and probe) and outputs: faulty or perfect. If faulty, it shows the region because of which it comes to the conclusion that is faulty.

b.) Show your results qualitatively in the report and write down your observation.

Question-4

![homography_transformation_example3](https://github.com/kwanit1142/Computer-Vision-on-different-scenarios/assets/54277039/db1811f5-cfc7-4d10-a767-73941abe4161)

What is a Homography Matrix? Write down the steps to copute Homography Matrix in detail with clear illustrative figures.

Question-5

![download](https://github.com/kwanit1142/Computer-Vision-on-different-scenarios/assets/54277039/b309a61a-1d23-4440-ab75-1b39d8be5b64)

What is stereo matching? Write down 3 applications of stereo matching. You can use web/books, but write the answer in your own words.

Question-6

![1_sMYUeozxiAaW5eHxn8KgOA](https://github.com/kwanit1142/Computer-Vision-on-different-scenarios/assets/54277039/b7bd65dd-9620-4862-a97f-faa700412ef7)

Write down steps to stitch images to create the panorama. Use the 3 Taj Mahal Images provided with this assignment to create one panaroma. Show panaroma into one.

## Lab-3 (SOTA Non-DL Methods)

Question-1

![train_faces](https://github.com/kwanit1142/Computer-Vision-on-different-scenarios/assets/54277039/b03943da-d3b0-4386-bf6f-d988097a0c67)

Use the subset of the LFW dataset provided with this assignment, include 1 face photograph of your favourite Indian sportsperson from the web to augment the dataset, and implement EigenFace Recognition from scratch. You may use the PCA library, but other functionalities should be originally written. Show top-K Eigen's Faces of the favourite Indian sportsperson you consindered in for different values of K. The report should also contain a detailed quantitative and qualitative analysis (Use provided data as train set and a test set will be provided separately).

Question-2

![0_BdetXYemwXwOqNTs](https://github.com/kwanit1142/Computer-Vision-on-different-scenarios/assets/54277039/ca8d1267-ae0f-413e-9085-10336d90ab5c)

Develop an Image Search Engine for CIFAR-10 that takes the image as a query and retrieves top-5 similar images using Visual BOW. Report Precision, Recall and AP. Draw the P-R Curve. Write down each step of implementation in clear and precise terms with an appropriate illustration.

Question-3

![0_cpIvTbyR-qaUgLx8](https://github.com/kwanit1142/Computer-Vision-on-different-scenarios/assets/54277039/8d6e828c-5307-482d-ae6c-0b711862a5ce)

Write down Viola Jone's Face detection steps in detail.

Question-4

![Detector-Model-Training-SVM-Classifier-with-the-HOG-Feature-for-Automobiles-and](https://github.com/kwanit1142/Computer-Vision-on-different-scenarios/assets/54277039/3466a5b1-d772-46fd-a334-a82e1f630160)

You are given a few deer train images with this assignment. Manually crop them to find out tight bounding boxes for Deer and also obtain some non-deer image patches of different sizes and aspect ratios. Compute HOG feeatures for deer and non-deer image patches and build an SVM classifier to classify deer v/s non-deer. Now, implement a sliding window object detection to find out the deer in the test images. Write down each step in the report. Also, objectively evaluate your detection performance.

## Lab-4 (Advanced Computer Vision Models)

Question-1

![1_XdCMCaHPt-pqtEibUfAnNw](https://github.com/kwanit1142/Computer-Vision-on-different-scenarios/assets/54277039/5fb1b040-e0f6-4bb5-a89c-b7d1fb660bbf)

Perform image classification using CNN on the MNIST dataset. Follow the standard train and test split. Design an 8-Layer CN Network (Choose your architecture, ex. filter size, number of channels, padding, activations, etc.). Perform the following tasks:-

a.) Show the calculation of output filter size at each layer of CNN.

b.) Calculate the number of parameters in your CNN. Calculation steps should be clearly shown in the report.

c.) Report the following on test data:- (Should be implemented from scratch)

1. Confusion Matrix
2. Overall and Classwise accuracy
3. ROC Curve (You can choose one class as positive and the rest classes as negative)

d.) Report Loss curve during training

e.) Replace your CNN with resnet18 and compare it with all metrics given in part 3. Comment on the final performance of your CNN and resnet18.

Question-2

![first2](https://github.com/kwanit1142/Computer-Vision-on-different-scenarios/assets/54277039/c4a3cb01-3669-4d23-88c6-c2f25682a937)

Download the Flicker8k dataset [images, captions]. Implement an encoder-decoder architecture for Image Captioning. For the encoder and decoder, you can use resnet/densenet/VGG and LSTM/RNN/GRU respectively. Perform the following tasks:-

a.) Split the dataset into train and test sets appropriately. You can further split the train set for validation. Train your model on the train set. Report loss curve during training.

b.) Choose an existing evaluation metric or propose your metric to evaluate your model. Specify the reason behind your selection/proposal of the metric. Report the final results on the test set.

Question-3

![download](https://github.com/kwanit1142/Computer-Vision-on-different-scenarios/assets/54277039/e5773596-2468-4ff2-b8fe-074bb571c6c9)

Use the dataset from Assignment-3 (Q4). Train YOLO object detection model (any version) on the train set. Compute the AP for the test set and compare the result with the HOG Detector. Show some visual results and compare both of the methods.
