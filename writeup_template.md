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

[image1]: ./results/histograms.jpg "Visualization"
[image2]: ./results/pipeline.jpg "Processing Pipeline"
[image3]: ./results/random_noise.jpg "Random Noise"
[image4]: ./my_signs/30lim1.jpg "Traffic Sign 1"
[image5]: ./my_signs/50lim1.jpg "Traffic Sign 2"
[image6]: ./my_signs/60lim1.jpg "Traffic Sign 3"
[image7]: ./my_signs/nopassing1.jpg "Traffic Sign 4"
[image8]: ./my_signs/novehicles1.jpg "Traffic Sign 5"
[image9]: ./my_signs/priorityway1.jpg "Traffic Sign 6"
[image10]: ./my_signs/round1.jpg "Traffic Sign 7"
[image11]: ./my_signs/school1.jpg "Traffic Sign 8"
[image12]: ./my_signs/stop1.jpg "Traffic Sign 9"
[image13]: ./my_signs/trafficsignals1.jpg "Traffic Sign 10"
[image14]: ./my_signs/turnleftahead1.jpg "Traffic Sign 11"
[image15]: ./my_signs/watchforpedestrians1.jpg "Traffic Sign 12"
[image16]: ./my_signs/wrongway1.jpg "Traffic Sign 13"
[image17]: ./my_signs/yield1.jpg "Traffic Sign 14"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used basic python and the numpy and pandas library to calculate summary statistics of the traffic signs data set:

* The size of training set is ```n_train = X_train.shape[0]```
* The size of the validation set is ``` n_validation = X_valid.shape[0]```
* The size of test set is ```n_test = X_test.shape[0]```
* The shape of a traffic sign image is ```image_shape = X_train.shape[1 : 4]```
* The number of unique classes/labels in the data set is ```len(np.unique(test['labels']))```

These yield the following results:
```
Number of training examples =  34799
Number of testing examples =  12630
Image data shape =  (32, 32, 3)
Number of classes =  43
```

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. 

After counting the classes, and transforming to a pandas dataframe with ```df = pd.DataFrame(categories_count, index = sign_names.values(), columns = ["Count"])```, I've used histograms to show how the data is balanced between each of the 43 classes. I display 2 images per set (train, validation, test), one with: ```df.plot(bins = 100, kind='hist')``` which shows how many samples exist per class in the dataset. The second, with ```df.hist(bins = 100, cumulative = True)``` is a cumulative histogram:

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step I increased the brightness of the images due to the fat that many had ver low lightness. I first tried to do a histgram equalization on the ```Y``` channel of the ```YUV``` converted image with ```img_yuv[ : , : , 0] = cv2.equalizeHist(img_yuv[ : , : , 0])``` but found the results unsatisfactory since some images that had very low brightness also had  very low range of brightnesses (lower 5%) which meant that brought to a proper rane were very pixelated. I then went back to a simple brightness increase of ```30``` on the ```HSV``` transform, with a clipping, with:

```
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(hsv)

lim = 255 - value
v[v > lim] = 255
v[v <= lim] += value

final_hsv = cv2.merge((h, s, v))
img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
```

At the end I do a normalization which is simply a division by ```255``` to bring the mean of the images around ```0``` and aid in the gradient descent. Original means as well as adjusted means are shown below (for all train, test and validation data):

```
print(np.mean(X_train_proc))
print(np.mean(X_test_proc))
print(np.mean(X_valid_proc))
...
82.677589037
82.1484603612
83.5564273756
...
0.417663127374
0.414353351674
0.419853065648
```

The end result shows that normalization has no visible effect (just aids in the algorithm) but brightness increase helps make images visible while not ruining the bright ones too much.

![alt text][image2]

I did not generate additional data as this wasn't required. Furthermore, from my peers' experience, forcing this easily helps you overfit on data, and you don't want that. Nevetheless, accuracy is good even without dataset augmentation.

I've tried grayscaling but it didn't work well for me. The notebook with that initial variant is still included.

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

I have experimented with multiple models after going online and searching for ideas. I started from LeNet (1), then I augmented it with an additional convolutional layer and an additional fully connected layer (2). 

I then tried to squash the fully-connected layers into a single one (3). This din't seem to add to the accuracy. 

I reverted to my own (2) and added dropout layers to avoid overfiting. I finally avoided overfitting through hyperparameter adjustment and left the final dropout layers with a 95% keep chance.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5 (6f) 	| 1x1 stride, VALID padding, outputs 28x28x6f 	|
| reLU					|												|
| Max pooling			| 2x2 stride,  outputs 14x14x6f 				|
| Convolution 3x3 (16f)	| 1x1 stride, VALID padding, outputs 12x12x16f	|
| reLU					|												|
| Max pooling	      	| 2x2 stride,  outputs 6x6x16f					|
| Convolution 3x3 (26f)	| 1x1 stride, VALID padding, outputs 4x4x26f	|
| reLU					|												|
| flatten				| 4x4x26f to 416								|
| Fully connected		| 416 to 250   									|
| Fully connected		| 250 to 120   									|
| Fully connected		| 120 to 84    									|
| Output				| 84 to 43     									|
|						|												|


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I several approaches, starting with the batch size, learning rate and epochs in the default file. I used the basic settings from the lessons (ADAM optimizer).

I got these results after training multiple times (validation accuracy):

*) LeNet (1) with default hyperparams and unprocessed dataset: 0.895, 0.883, etc
*) LeNet (1) with efault hyperparams and just normalization: 0.906, 0.899, 0.909, etc
*) My modified LeNet (2) with no dropout, normalization: 0.915, 0.922, 0.919, 0.914, etc
*) Flattened LeNet (3) with no dropout, normalization: 0.901, 0.905, 0.910, etc
*) Final LeNet (2) with improved processing: 0.924, 0.931, 0.929, etc.
*) Final LeNet (2), improved processing, 20 epochs, 0.0015 learning rate: 0.930, 0.937, 0.928, etc.
*) Final LeNet (2), improved processing, with dropout 0.5: 0.868, 0.873, etc
*) Final LeNet (2), improved processing, batch_size 64, learning_rate 0.002, 30 epochs: 0.917, 0.944, 0.950, 0.932, 0.919, etc. (shaky!)
*) Final LeNet (2), improved processing, batch_size 64, learning_rate 0.001, 100 epochs, 0.927, 0.934, etc.
*) Final LeNet 92), best processing, batch_size 64, learning_rate 0.0012, 30 epochs, 0.95 dropout: 0.934, 0.939, 0.944, 0.941, 0.958, etc.

I feel that many can "cheat" by using a big learning rate and getting the training to stop when it accidentally hits large values (such as 0.95), but as I've demonstrated in the notebook, this approach does not give good results on the actual test set and real images, not better than more training with less learning rate anyway. I would go on with a learning_rate of about 0.0075 and 200 epochs, with a bit more preprocessing and some augmentation (but not much) and it's likely that accuracy could go north of 0.96-0.97.

I've saved my best models in the respective directories.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 93.4%
* validation set accuracy of 91%
* my test set accuracy of about 93%

If an iterative approach was chosen:

As outlined in the chapter above, I tried to work my way up from the basic LeNet. No matter what I tried I couldn't get my accuracy higher than about 0.92. 

Then I made it more complex (able to infer more) by adding extra layers (first a convolutional layer, then a fully-connected one). As Andrew Ng and others suggest, too many parameters in your network can lead to overfitting which is what I saw initially in the discrepancies between training and validation (0.93 to 0.89 accuracies!). I've seen others go to differences of 10% between the two. It's no point in getting your validation accuracy so high when on real world data you fail. 

So then I tried to add dropout and let one single fully-connected layer do the job, but the combinations lead to no significant accuracy increases, or perhaps they were masked by the hyperparameters.

After considering that I have done a sufficient job (dropout layers to 95% keepprob added as final step) and getting 0.93 and 0.94 training accuracies (highly variable), I decided to start adjusting the hyperparameters (initially sticking with 10-20 epoch.

I first went to a higher learning_rate which sometimes led to good outcomes (0.0015 and 0.002 were attempted at first) leading to a single 0.951 accuracy and plenty of 0.92-0.94 ones, but the final test accuracies were pretty low (0.85 or even less). This approach wasn't sound. I then tried to do a very low learning_rate (going as low as 0.0007) but the system stabilized even after 100 epochs at around 0.92 accuracy, too low for the requirements of the project. 

I then changed the batch_size up and down, going as low as 32 (which slowed down training a lot) and as high as 256 (with erratic training results) settling on an optimum of 64.

With the final results aand increasing the epochs to about 30 I arrived at a steady training accuracy of 0.93-0.94. My own set of about 15 images held a 93% accuracy as well and the validation stands at about 91% which means that there isn't much overfitting, and the results hold for real images. This means he model is relatively stable.

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are 14 German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8] ![alt text][image9]
![alt text][image10] ![alt text][image11] ![alt text][image12]
![alt text][image13] ![alt text][image14] ![alt text][image15]
![alt text][image6] ![alt text][image17]

Of these, the one more dificult to classify is the "sign" one since after being made small it's a blur. 

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Yield					| Yield											|
| Pedestrians  			| Pedestrians									|
| 30 km/h	      		| 30 km/h   					 				|
| 50 km/h	      		| 50 km/h   					 				|
| 60 km/h	      		| 30 km/h   					 				|
| Stop Sign      		| Stop sign   									| 
| Priority Road			| Priority Road      							|
| School     			| School 										|
| Traffic Signals		| Traffic Signals      							|
| No Entry     			| No Entry 										|
| No Vehicles  			| No Vehicles									|
| Roundabout  			| Roundabout									|
| No Passing  			| No Passing 									|
| Turn Left Ahead 		| Turn Left Ahead								|


The model was able to correctly guess 13 of the 14 traffic signs, which gives an accuracy of 93%. This compares favorably to the accuracy on the test set of 93%.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The top five soft max probabilities were:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0					| Yield											|
| 1.0  					| Pedestrians									|
| 0.99					| 30 km/h   					 				|
| 0.87					| 50 km/h   					 				|
| 0.53					| 60 km/h (wrongly predicted as 30 km/h 		|
| 1.0 					| Stop sign   									| 
| 1.0 					| Priority Road      							|
| 1.0 					| School 										|
| 0.75 					| Traffic Signals      							|
| 1.0 					| No Entry 										|
| 1.0  					| No Vehicles									|
| 1.0  					| No Passing									|
| 0.97 					| Roundabout									|
| 1.0 					| Turn Left Ahead								|


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


