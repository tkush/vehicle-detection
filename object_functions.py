import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import sys

from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
from scipy.ndimage.measurements import label

class Get_Features(object):

    @staticmethod
    #  Define a function to return HOG features and visualization
    def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
        if vis == True:
            features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                    cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False, 
                                    visualise=True, feature_vector=False)
            return features, hog_image
        else:      
            features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                        cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False, 
                        visualise=False, feature_vector=feature_vec)
            return features

    @staticmethod
    # Define a function to compute binned color features  
    def bin_spatial(img, size=(32, 32)):
        return (cv2.resize(img, size).ravel())

    @staticmethod
    # Define a function to compute color histogram features  
    def color_hist(img, nbins=32, bins_range=(0, 256)):
        # Compute the histogram of the color channels separately
        channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
        channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
        channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
        # Concatenate the histograms into a single feature vector
        hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
        # Return the individual histograms, bin_centers and feature vector
        return hist_features

    @staticmethod
    # Function to display a simple progress bar
    def drawProgressBar(percent, barLen = 20, text=""):
        sys.stdout.write("\r")
        progress = ""
        for i in range(barLen):
            if i < int(barLen * percent):
                progress += "="
            else:
                progress += " "
        msg = text + "[ %s ] %.2f%%" % (progress, percent * 100)
        sys.stdout.write(msg)
        sys.stdout.flush()

    @staticmethod
    # Define a function to extract features from a list of images
    # Have this function call bin_spatial() and color_hist()
    def extract_features(single_img, imgs, cspace='RGB', spatial_size=(32, 32),
                            hist_bins=32, hist_range=(0, 256), orient=9,
                            pix_per_cell=8, cell_per_block=2, hog_channel=0,
                            spat_feat=False, hist_feat=False, hog_feat=False):
        # Create a list to append feature vectors to
        features = []
        total = len(imgs)
        for idx,img in enumerate(imgs):
            if (single_img):
                image = imgs[0]
            else:
                # Read in image
                image = cv2.imread(img)
            
            # apply color conversion if other than 'RGB'
            if cspace != 'RGB':
                if cspace == 'HSV':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                elif cspace == 'LUV':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2LUV)
                elif cspace == 'HLS':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
                elif cspace == 'YUV':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
                elif cspace == 'YCrCb':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
            else: 
                feature_image = np.copy(image)      
            # Apply bin_spatial() to get spatial color features
            if ( spat_feat ):
                spatial_features = Get_Features.bin_spatial(feature_image, size=spatial_size)
                # features.append(spatial_features)
            # Apply color_hist() to get histogram features
            if ( hist_feat ):
                hist_features = Get_Features.color_hist(feature_image, nbins=hist_bins, bins_range=hist_range)
            # Call get_hog_features with vis=False, feature_vec = True
            if ( hog_feat ):
                if hog_channel == 'ALL':
                    hog_features = []
                    for channel in range(feature_image.shape[2]):
                        hog_features.append(Get_Features.get_hog_features(feature_image[:,:,channel], 
                                            orient, pix_per_cell, cell_per_block, 
                                            vis=False, feature_vec=True))
                    hog_features = np.ravel(hog_features)        
                else:
                    hog_features = Get_Features.get_hog_features(feature_image[:,:,hog_channel], orient, 
                                pix_per_cell, cell_per_block, vis=False, feature_vec=True)
            features.append(np.concatenate((spatial_features, hist_features, hog_features)))

            if (single_img==False):
                Get_Features.drawProgressBar(float((idx+1)/total), barLen = 20, text="Extracting features...")
        # print(len(features))
        # Return list of feature vectors
        if (single_img==False):
            Get_Features.drawProgressBar(1.0, barLen = 20, text="Extracting features...")
            print()
        return features

    # Define a function that takes an image,
    # start and stop positions in both x and y, 
    # window size (x and y dimensions),  
    # and overlap fraction (for both x and y)
    @staticmethod
    def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None], 
                        xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
        # If x and/or y start/stop positions not defined, set to image size
        if x_start_stop[0] == None:
            x_start_stop[0] = 0
        if x_start_stop[1] == None:
            x_start_stop[1] = img.shape[1]
        if y_start_stop[0] == None:
            y_start_stop[0] = 0
        if y_start_stop[1] == None:
            y_start_stop[1] = img.shape[0]
        # Compute the span of the region to be searched    
        xspan = x_start_stop[1] - x_start_stop[0]
        yspan = y_start_stop[1] - y_start_stop[0]
        # Compute the number of pixels per step in x/y
        nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
        ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
        # Compute the number of windows in x/y
        nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
        ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
        nx_windows = np.int((xspan-nx_buffer)/nx_pix_per_step) 
        ny_windows = np.int((yspan-ny_buffer)/ny_pix_per_step) 
        # Initialize a list to append window positions to
        window_list = []
        # Loop through finding x and y window positions
        # Note: you could vectorize this step, but in practice
        # you'll be considering windows one by one with your
        # classifier, so looping makes sense
        for ys in range(ny_windows):
            for xs in range(nx_windows):
                # Calculate window position
                startx = xs*nx_pix_per_step + x_start_stop[0]
                endx = startx + xy_window[0]
                starty = ys*ny_pix_per_step + y_start_stop[0]
                endy = starty + xy_window[1]
                
                # Append window position to list
                window_list.append(((startx, starty), (endx, endy)))
        # Return the list of windows
        return window_list

    # Define a function to draw bounding boxes
    @staticmethod
    def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
        # Make a copy of the image
        imcopy = np.copy(img)
        # Iterate through the bounding boxes
        for bbox in bboxes:
            # Draw a rectangle given bbox coordinates
            cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
        # Return the image copy with boxes drawn
        return imcopy

    # Define a function you will pass an image 
    # and the list of windows to be searched (output of slide_windows())
    @staticmethod
    def search_windows(img, windows, clf, scaler, color_space='RGB', 
                        spatial_size=(32, 32), hist_bins=32, 
                        hist_range=(0, 256), orient=9, 
                        pix_per_cell=8, cell_per_block=2, 
                        hog_channel=0, spatial_feat=True, 
                        hist_feat=True, hog_feat=True):

        #1) Create an empty list to receive positive detection windows
        on_windows = []
        #2) Iterate over all windows in the list
        for window in windows:
            #3) Extract the test window from original image
            test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))      
            #4) Extract features for that window using single_img_features()
            features = Get_Features.extract_features(True, [test_img], cspace=color_space, 
                                spatial_size=spatial_size, hist_bins=hist_bins, 
                                orient=orient, pix_per_cell=pix_per_cell, 
                                cell_per_block=cell_per_block, 
                                hog_channel=hog_channel, spat_feat=spatial_feat, 
                                hist_feat=hist_feat, hog_feat=hog_feat)
            #5) Scale extracted features to be fed to classifier
            test_features = scaler.transform(np.array(features).reshape(1, -1))
            #6) Predict using your classifier
            prediction = clf.predict(test_features)
            #7) If positive (prediction == 1) then save the window
            if prediction == 1:
                on_windows.append(window)
        #8) Return windows for positive detections
        return on_windows

    @staticmethod
    # Define a function to extract HOG features from the entire image of interest
    def find_cars_my(src_img, draw_on_img, dict_of_params, window_sizes, y_lims):
        img_copy = src_img.copy()
        found_bbox_list = []
        
        # heat_img = np.zeros_like(src_img[:,:,0]).astype(np.float)
        
        # Get parameters
        orient = dict_of_params["param"]["orient"]
        pix_per_cell = dict_of_params["param"]["pix_per_cell"]
        cell_per_block = dict_of_params["param"]["cell_per_block"]
        spatial_size = dict_of_params["param"]["spatial_size"]
        hist_bins = dict_of_params["param"]["hist_bins"]
        X_scaler = dict_of_params["X_scaler"]
        svc = dict_of_params["classifier"]

        n_blocks_per_window = 64 // pix_per_cell - 1    
        prev_size = None
        for idx, size in enumerate(window_sizes):
            cropped_img = img_copy[y_lims[idx][0]:y_lims[idx][1],:,:]
            cspace_img = Get_Features.convert_color(cropped_img,'BGR2YCrCb')
            
            if ( prev_size != size ):
                # Scale image based on the size of the window as compared to 64x64
                # Example: window size is 128x128, scale the image down by a factor of 128/64 = 2
                # Example: window size is 32x32, scale the image down by a factor of 32/64 = 0.5
                scale = size / 64
                imshape = cspace_img.shape
                cspace_img = cv2.resize(cspace_img, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))

                # Split into channels
                ch1 = cspace_img[:,:,0]
                ch2 = cspace_img[:,:,1]
                ch3 = cspace_img[:,:,2]

                # Compute individual channel HOG features for the entire image
                # This is 37 x 159 x ravel
                hog1 = Get_Features.get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
                hog2 = Get_Features.get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
                hog3 = Get_Features.get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

                prev_size = size
                
            # Get a list of the sliding windows for the scaled image
            window_list = Get_Features.slide_window(cspace_img, [None, None], [None, None], (64, 64), (0.75, 0.75))
            
            for window in window_list:
                startx = window[0][0]
                starty = window[0][1]
                endx = window[1][0]
                endy = window[1][1]
                y_idx_hog_start = starty // pix_per_cell
                y_idx_hog_end = y_idx_hog_start + n_blocks_per_window - 1
                x_idx_hog_start = startx // pix_per_cell
                x_idx_hog_end = x_idx_hog_start + n_blocks_per_window - 1
                pixels_per_block = 16
                skip_cells = 2
                # print("Y: Start HI: {0} End HI: {1}".format(y_idx_hog_start,y_idx_hog_end))
                # print("X: Start HI: {0} End HI: {1}".format(x_idx_hog_start,x_idx_hog_end))
                # Extract HOG for this patch
                hog_feat1 = hog1[y_idx_hog_start:y_idx_hog_end+1, x_idx_hog_start:x_idx_hog_end+1].ravel()            
                # print("Raveled:{}".format(hog_feat1.shape)) 
                hog_feat2 = hog2[y_idx_hog_start:y_idx_hog_end+1, x_idx_hog_start:x_idx_hog_end+1].ravel() 
                hog_feat3 = hog3[y_idx_hog_start:y_idx_hog_end+1, x_idx_hog_start:x_idx_hog_end+1].ravel() 
                hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
                
                # Extract the image patch and resize to match that of training data
                subimg = cspace_img[starty:endy, startx:endx]
            
                # Get color features
                spatial_features = Get_Features.bin_spatial(subimg, size=spatial_size)
                hist_features = Get_Features.color_hist(subimg, nbins=hist_bins)

                # Print out size of features extracted
                # print("Extracted {} spatial features".format(spatial_features.shape))
                # print("Extracted {} hist features".format(hist_features.shape))
                # print("Extracted {} hog features".format(hog_features.shape))
                features_stacked = np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1)
                # print("Extracted {} consolidated features".format(features_stacked.shape))
                
                # Scale features and make a prediction
                # print(X_scaler.get_params())
                test_features = X_scaler.transform(features_stacked)    
                test_prediction = svc.predict(test_features)
                
                if test_prediction == 1:
                    startx_scaled = np.int(startx*scale)
                    starty_scaled = y_lims[idx][0] + np.int(starty*scale)
                    win_size_scaled = np.int(64*scale)
                    found_bbox_list.append(((startx_scaled, starty_scaled),(startx_scaled+win_size_scaled,starty_scaled+win_size_scaled)))
                    # cv2.rectangle(draw_on_img,(startx_scaled, starty_scaled),(startx_scaled+win_size_scaled,starty_scaled+win_size_scaled),(0,255,0),1) 
                    # cv2.imshow("",draw_on_img)
                    # cv2.waitKey(5)
        return found_bbox_list
    
    @staticmethod
    def convert_color(img, conv='RGB2YCrCb'):
        if conv == 'RGB2YCrCb':
            return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        if conv == 'BGR2YCrCb':
            return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        if conv == 'BGR2LUV':
            return cv2.cvtColor(img, cv2.COLOR_BGR2LUV)

    @staticmethod
    def add_heat(heatmap, bbox_list):
        for box in bbox_list:
            heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
            # print(heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]])

        # Return updated heatmap
        return heatmap

    @staticmethod
    def apply_threshold(heatmap, thres):
        heatmap[ heatmap <= thres ] = 0
        # heatmap[ heatmap >= thres ] = 1
        return heatmap

    @staticmethod
    def draw_labeled_bboxes(img, labels):
        # Iterate through all detected cars
        for car_number in range(1, labels[1]+1):
            # Find pixels with each car_number label value
            nonzero = (labels[0] == car_number).nonzero()
            # Identify x and y values of those pixels
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            # Define a bounding box based on min/max x and y
            bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
            # Draw the box on the image
            cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
        # Return the image
        return img