# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:

* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./figures/rand_img_label.png "Random Image"
[image2]: ./figures/labels_hist.png "Label Histogram"
[image3]: ./figures/grayscale.png "Grayscale"
[image4]: ./figures/augmentation.png "Data Augmentation"
[image5]: ./figures/test_errors.png "Test Errors"
[image6]: ./figures/new_signs.png "New Traffic Signs"
[image7]: ./figures/new_predictions.png "New Traffic Signs Predictions"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is `(32, 32, 3)`
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

First, I manually examined a random image from the training dataset and made sure the image and its label match.
![alt text][image1]

Next, I compared the label distribution among train, validation and test datasets:
![alt text][image2]
From the histogram above, the label distributions are very similar across these three datasets. However, the classes are not balanced. For example, there are more examples of classes `1` (Speed limit (30km/h)) and `2` (Speed limit (50km/h)) than that of class `19` (Dangerous curve to the left).

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

I applied the three pre-processing steps to the dataset: data augmentation, feature normalization and grayscaling. Below I'll provide the rationale and example output for each of these steps:

* **Feature normalization**: I only performed the basic feature normalization for this project: `(pixel - 128)/ 128`. The reason for doing feature normalization is to improve the conditioning of training data and speed up optimization.
* **Grayscale**: I think there are pros and cons for converting images to grayscale:
	* Pros: more robust against variations in color saturation and lightness.
	* Cons: loss of valuable information. For example, turn signs are normally blue, while stop signs are red. This color difference would be useful in classification but it's lost after grayscaling.
	* I used grayscale in training because, empirically on the traffic sign dataset, it leads to faster convergence and similar accuracy than RGB images.
	* ![alt text][image3]
* **Data augmentation**: As I'll discuss more in the sections below, the LeNet-5 architecture shown in the classroom lead to overfitting on the traffic sign dataset (Train accuracy 99% vs. Validation accuracy 89%). Increasing the training data size is an effective means to deal with overfitting. Therefore I tried to augment the original training data with synthetic examples, including:
	* Rotation of a random degree between `-20` and `20`
	* Translation along both dimensions (random number of pixels from `-4` to `4`)
	* Gaussian noise is added which visually lead to darker images

	Therefore, for each image in the training dataset I added three new synthetic images, and this enlarges the training dataset 3 folds. Below are the illustration of synthetic images from 3 random images originally in the dataset.

	* ![alt text][image4]

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 grayscale image   							| 
| Convolution_1 5x5     	| 1x1 stride, valid padding, outputs 28x28x6	|
| RELU_1					|												|
| Max\_pooling\_1	      	| 2x2 stride,  outputs 14x14x6				|
| Convolution_2 5x5	    | 1x1 stride, valid padding, outputs 10x10x16					|
| RELU_2					|												|
| Max\_pooling\_2	      	| 2x2 stride,  outputs 5x5x16
    |
| 4 Fully connected (with Dropout) | Size: 400 -> 120 -> 84 -> 43						|
| Softmax				|        			
 

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used:

* Optimizer: `tf.train.AdamOptimizer`
* Batch size: `128`
* Number of epochs: `100`
* Dropout: `keep_prob = 0.8`
* Learning rate: `0.001`

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:

* training set accuracy of `0.997`
* validation set accuracy of `0.983`
* test set accuracy of `0.959`

If an iterative approach was chosen:

* **Starting point**: I started with the LeNet-5 architecture shown in the classroom and applied it on RGB images. The training accuracy can reach `0.98` while the validation accuracy was only `0.89` in `10` epochs, which suggests overfitting. To solve this, I planned to use Dropout and Data Augmentation.
* **Dropout**: I added dropout layers after each of the fully connected layers and made the `keep_prob` a tunable parameter. With `keep_prob = 0.5`, I was able to achieve validation accuracy of `0.93` in `10` epochs.
* **Data augmentation**: Adding additional synthetic data further improved the validation accuracy to `0.95` in `10` epochs. With more data, I also lowered the dropout strength. After changing `keep_prob` to `0.8`, validation accuracy became `0.96`.
* **Grayscaling**: The pros and cons of converting images to grayscale were discussed above. In this specific task, using grayscale images has led to faster convergence and similar accuracy in the end.
* **Epoch**: After finalizing the training setup from above, I finally ran the optimization for 100 epochs and saved the model with the best validation accuracy (`0.983`).

The final model achieved a test accuracy of `0.959`. I looked at some misclassified examples as shown below. The first columns are random examples of misclassified test images.The remaining columns are the top 5 prediction from the model.
![alt text][image5]

 It seems that the misclassifications are caused by the following reason:

 * Loss of color information: this is probably the case for the 1st, 3rd and 4th images
 * Traffic sign hidden behind other objects: this is probably the case for the 4th image
 * Dark images: this probably caused the misclassification for the 5th image

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Below are five German traffic signs that I found on the web. I deliberately chose images that are similar to the ones in training dataset for better generalization.

![alt text][image6]

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The model correctly predicted all of the test images right (100% accuracy). This is because the test images I chose are very similar to the ones from the training dataset. If the test images are different (i.e., from a different distribution), the model may not be able to generalize well.


#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

Below are the top 5 predictions and the probabilities for these new images. The model was able to correctly classify all of them with high confidence because the test images I chose are very similar to the ones from the training dataset.
![alt text][image7]

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


