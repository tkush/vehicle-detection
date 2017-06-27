# vehicle-detection
Vehicle detection in dash cam video

[![Vehicle detection](https://img.youtube.com/vi/ATBe8aiQ8xo/0.jpg)](https://www.youtube.com/watch?v=ATBe8aiQ8xo "Vehicle detecttio")

The goals of this code are the following:
* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Apply a color transform and append binned color features, as well as histograms of color, to the HOG feature vector.
* Normalize the features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run the pipeline on a video stream and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

### HISTOGRAM OF ORIENTED GRADIENTS (HOG) 
HOG gradients were made popular by the work done by Dalal and Triggs, researchers at the INRIA in France. HOG descriptors were shown to be very good at object detection when used as feature descriptors. They were shown to perform very well while detecting human figures in images, and in this project, they are successfully used to detect vehicles (specifically passenger vehicles) in video captured from a moving car. 
In this work, the hog function made available in the skimage module (skimage.feature.hog) is used to determine HOG features given an input image and input parameters. This is implemented within the static function get_hog_features defined in the file object_function.py within the Get_Features class:
``` python
    @staticmethod
    #  Define a function to return HOG features and visualization
    def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
```

The input to this function is:

`img`,	Array (numpy), This represents the input image which is an individual frame of the video. 
`orient`,	Integer,	A number that represents the number of orientation bins for the HOG feature extraction
`pix_per_cell`,	Integer,	A number that represents the number of pixels contained per cell for the HOG feature extraction
`cell_per_block`,	Integer,	A number that represents the number of cells per block for the HOG feature extraction
`vis`,	Bool,	A True/False variable that determines if the result of the HOG feature extraction should be visualized as an image. Setting this to TRUE makes this function return a 2-tuple
`feature_vec`,	Bool,	A True/False variable that determines whether an array of the feature vector should be returned as opposed to the raw features themselves

The function get_hog_features is used to extract HOG features in several parts of the code. Here is an example of an input image (in YCrCb colorspace), and it’s corresponding HOG feature descriptor:
 	 	 	 
![HOG](/images/hog.png)

Different color spaces can be used for the input image to this function – RGB, HSV, HLS, YCrCb etc. 

#### Choosing parameters for the HOG feature extraction
To determine the choice of parameters for the HOG feature extractor and the color space for the input image, several iterations were run to The YCrCb was used in the entirety of this project based on the accuracy of the classifier. This is explained further in the next section.

### TRAINING THE CLASSIFIER
#### PREPARING THE DATA
The GTI and KITTI data sets provided as part of the Udacity project were used as training data to train a linear **Support Vector Machine (SVM)** classifier. The `LinearSVC` module from within the `sklearn.svm` library is used for this purpose. The classifier predicted was trained to predict each image in the training data set as a car or not a car. 
The GTI and KITTI data sets provide labelled data for both cars and non-cars. Here are some examples from the data set: 
    	    
![Cars	Non-Cars](/images/cars_notcars.png)

Further, **hard negative mining** was applied by extracting false positives from the prediction on the project video and these were used to re-train the classifier to reduce the number of false positives. An image with false positives is shown below: 
 
![An image from the video showing a false positive (in green box)](/images/falsepos.png)

Some images used for the hard-negative mining are shown here: 

![Images used for hard negative mining](/images/hardnegmining.png)

The positive data (cars) is subsampled such that `N(cars) << N(not_cars)` in the training dataset. The idea behind this reduction for the positive data is that the classifier is less likely to have a bias towards predicting an image as a car. In this project, there are **5203** car images and **9694** non-car images for a total of **14,897** training images.

### EXTRACTING FEATURES FOR TRAINING THE CLASSIFIER
The features used to train the classifier are: 
1.	Spatial binning features: The input image is resized to (n x n) and the feature vector is obtained by use of the ravel() function
2.	Color histogram features: A color histogram of the input image is calculated using a certain number of bins for each channel of the input image. These individual histogram features for each channel are then simply concatenated 
3.	HOG features: This is as defined in the previous section
The parameters for the feature extraction are:
1.	Color space for the input image
2.	Number of orientation bins for the HOG extractor
3.	Pixels per cell for the HOG extractor
4.	Cells per block for the HOG extractor
5.	Channel number used to extract HOG features from the input image (0, 1, 2 or ALL)
6.	Size of the spatial bin 
7.	Number of histogram bins
The choice of these parameters is determined by looking at the accuracy of the trained classifier on the held-out test data for varying parameter values. The parameters that yield the highest accuracy for the classifier are chosen for this project. 
```python
    param["color_space"] = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    param["orient"] = 9  # HOG orientations
    param["pix_per_cell"] = 8 # HOG pixels per cell
    param["cell_per_block"] = 2 # HOG cells per block
    param["hog_channel"] = "ALL" # Can be 0, 1, 2, or "ALL"
    param["spatial_size"] = (4, 4) # Spatial binning dimensions
    param["hist_bins"] = 32    # Number of histogram bins
    param["spatial_feat"] = True # Spatial features on or off
    param["hist_feat"] = True # Histogram features on or off
    param["hog_feat"] = True # HOG features on or off
```

Using these parameters, the size of the feature vector for each training image is **5436**. Feature extraction for the dataset takes **~265s (or 4 minutes 25s)** on an Intel i7 CPU @ 2.7GHz running on a single core. 
Before starting to train the classifier, a random 80-20 split is done on the complete data where 80% of the randomly shuffled data is used for training and the remaining 20% is held out for testing. Further, scaling is performed on the entire data using the `StandardScaler()` method within the `sklearn.preprocessing` module.
The classifier trains in about 57s yielding an accuracy of **~99%** on the test set. This means there are about **148** incorrect predictions which may have an effect when running on the project video.
To speed up predictions, the classifier and the classifier parameters are pickled and stored on disk.

### SLIDING WINDOW SEARCH
To detect cars within an image frame of the project video, a sliding window approach is used. This is briefly outlined below: 

Step 1.	Choose a size for the sliding window

Step 2.	Crop the test image to this sliding window and extract features for this window. Predict using the trained classifier as to whether this is a car or not

Step 3.	“Slide” this window in both X and Y directions within the image with some amount of overlap

Step 4.	Repeat Step 2 until the entire region of interest is covered

Step 5.	Repeat from Step 1 

This approach is implemented in the function `Get_Features.slide_window` which prepares a list of windows to be used for feature extraction. Since the HOG features are time consuming to extract for each window individually, the following overall approach is used to find vehicles in the input image: 

Step 1.	The image is cropped to within the region of interest. This ROI varies for each sliding window size. For example, a smaller sliding window is useful to detect a car that appears smaller in size in the image. Such a car is closer to the vanishing point in the image and thus the ROI for such a sliding window is smaller. 

Step 2.	The input image is converted to the same color space as the images in the training dataset

Step 3.	The input image is scaled such that a window of size n by n becomes a window of size 64 x 64 in the scaled image. This is done since the training data is of size 64 x 64, therefore the test data must match this for the prediction to work reliably

Step 4.	HOG features are obtained for the entire scaled image (for all channels) only once

Step 5.	A sub-sampling scheme is used to obtain HOG features for each sliding window. 

Step 6.	Spatial and color histogram features are obtained

Step 7.	The classifier is used to predict whether the contents of the window contain a car or not

Step 8.	A 75% sliding window overlap is used since this seemed to provide the best results while testing

Step 9.	The window then “slides” across the length and breadth of the ROI till the entire image is covered

Step 10.	Repeat from step 3.

The following images show the ROI and the sliding windows used for this project:
 	 
![Window sizes](/images/windows.png)
 	 
**Note**: In the images above, the windows appear to be smaller than actual size since they are overlapping 75% of one another in both X and Y directions. 

### DEALING WITH FALSE POSITIVES
Two techniques are used to remove false positives from the results obtained using the process above: 
1.	As mentioned before, hard negative mining was used 
2.	A thresholding on a “heat map” is also used to remove false positives. The idea is to consider only those areas of the image that have more than 2 positives indicated (of different window sizes). As an example, consider the image below. The red box shows the bounding box that is accepted as a true positive i.e. where more than 2 green boxes overlap. 
 
![Thresholding on a “heat” map](/images/thres.png)

Here are some more images that show the results from the pipeline. The thin green boxes show car detections from the classifier and the thicker green/red boxes show car detections after removing false positives: 
 	 
![res](/images/out1.png)

![res](/images/out2.png)
 	 
### VIDEO PIPELINE
The output video is uploaded here

[![Vehicle detection](https://img.youtube.com/vi/ATBe8aiQ8xo/0.jpg)](https://www.youtube.com/watch?v=ATBe8aiQ8xo "Vehicle detection")

### DEALING WITH FALSE POSITIVES
For each frame, multiple bounding boxes are combined using the heat map thresholding as described in the previous section. Sometimes, there are peaks in the heat map thresholding that are not connected to one another. These small peaks are merged in to the larger peaks based on a threshold value. This is shown below:
 
![After heat map thresholding](/images/merge.png)

In addition to this, a simple overlapping box test is used between two consecutive video frames to determine the amount of overlap between the two bounding boxes:

![Any bounding boxes that have an overlap below a certain threshold are dropped](/images/overlap.png)

### DISCUSSION
#### CHALLENGES
* Dealing with false positives took the most amount of time in this project. Since the classifier is trained on images that do not appear in the project video, there are several false positives that appear. Hard negative mining takes care of some of this – but this was found to be time consuming
* Debugging the code was also time consuming since producing the output frame takes some seconds per frame. An attempt was made to mitigate this using parallel processing, however, this was found to be detrimental to the solution. This is discussed next. 
* This pipeline was tuned for the project video. It will likely fail in the following scenarios: 
  - The camera within the car is mounted differently such that the horizon is at a different level
  -	For windy roads, there may be background that is picked up within the ROI instead of the road itself i.e. the ROI will need to be       changed
  - The classifier may perform differently under different conditions of light/rain/road etc. 

### ATTEMPT AT PARALLELIZING THE SOLUTION
To make an attempt at detecting vehicles faster, one approach that was tried was to divide the work between multiple processes. For this, the `Queue()` and `Process()` classes are used from the multiprocessing module within Python. The idea is to split the work done for detecting vehicles using multiple window sizes into 4 processes. Each process handles one or more window sizes in parallel and deposits it’s result onto the shared `Queue()` which is then processed together later. The motivation behind this is to reduce the amount of time taken in feature extraction by having parallel workers do the job. Here are some times for detecting vehicles using different window sizes (calculated serially): 

Window size (Time to detect)

192 (0.4s)

128 (0.9s)

96 (1.2s)

64 (2.84s)

32 (2.6s)

However, when this was run in parallel, the compute times were larger than the compute times for serial processing – likely due to one or both factors below: 
* Overhead time (pooling, de-pooling etc.) is significantly larger than time savings
* There is a bug in the implementation of the parallel code
Nonetheless, the implementation of the parallel code is also included in the submission for review. As a next step, I intend to profile the code more thoroughly and find out how the time taken for processing each frame can be reduced. 

### TO-DO
- [ ] Profile Python code to find bottleneck
- [ ] Re-write code in C++ to improve speed
- [ ] Combine lane finding and vehicle detection
- [ ] Investigate use of Faster-RCNN or MASK-RCNN for object detection
