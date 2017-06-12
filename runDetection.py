import glob
from object_functions import *
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
import time
import pickle
import os
from frame import *

generate_new = False
if __name__ == '__main__':
    if (generate_new):
        # Load the data
        images_notcar = glob.glob('data_large/non_vehicles/*/*.png')
        images_notcar_negmining = glob.glob('neg_mining/neg_examples/*.png')
        images_car = glob.glob('data_large/vehicles/*/*.png')
        images_car_posmining = glob.glob('neg_mining/pos_examples/*.png')
        cars = []
        notcars = []

        for image in images_notcar:
            notcars.append(image)
        for image in images_notcar_negmining:
            notcars.append(image)
        for image in images_car:
            cars.append(image)

        # Randomly pick half of car examples
        cars_chosen = random.sample(cars,len(notcars)/2)
        for image in images_car_posmining:
            cars_chosen.append(image)
        cars = cars_chosen
        
        print("Loaded data...")
        print("Found {} car images".format(len(cars)))
        print("Found {} non-car images".format(len(notcars)))

        # Extract features
        # Parameters
        param = {}
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

        start_time = time.time()
        car_features = Get_Features.extract_features(False, cars, cspace=param["color_space"], 
                                spatial_size=param["spatial_size"], hist_bins=param["hist_bins"], 
                                orient=param["orient"], pix_per_cell=param["pix_per_cell"], 
                                cell_per_block=param["cell_per_block"], 
                                hog_channel=param["hog_channel"], spat_feat=param["spatial_feat"], 
                                hist_feat=param["hist_feat"], hog_feat=param["hog_feat"])
        notcar_features = Get_Features.extract_features(False, notcars, cspace=param["color_space"], 
                                spatial_size=param["spatial_size"], hist_bins=param["hist_bins"], 
                                orient=param["orient"], pix_per_cell=param["pix_per_cell"], 
                                cell_per_block=param["cell_per_block"], 
                                hog_channel=param["hog_channel"], spat_feat=param["spatial_feat"], 
                                hist_feat=param["hist_feat"], hog_feat=param["hog_feat"])

        print("Computed",len(car_features)," car features of size ",len(car_features[0]))
        print("Computed",len(notcar_features)," not car features of size ",len(notcar_features[0]))
        print("Feature extraction took {} seconds".format(round(time.time() - start_time,2)))
        print()
        X = np.vstack((car_features, notcar_features)).astype(np.float64)                        
        # Fit a per-column scaler
        X_scaler = StandardScaler().fit(X)
        # Apply the scaler to X
        scaled_X = X_scaler.transform(X)

        # Define the labels vector
        y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

        # Split up data into randomized training and test sets
        rand_state = np.random.randint(0, 100)
        X_train, X_test, y_train, y_test = train_test_split(
            scaled_X, y, test_size=0.2, random_state=rand_state)

        print('Using:',param["orient"],'orientations',param["pix_per_cell"],
            'pixels per cell and', param["cell_per_block"],'cells per block')
        # Use a linear SVC 
        svc = LinearSVC()
        # Check the training time for the SVC
        t=time.time()
        svc.fit(X_train, y_train)
        t2 = time.time()
        print(round(t2-t, 2), 'Seconds to train SVC...')
        # Check the score of the SVC
        print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))

        # Save params and other stuff to file
        saved_dict = {}
        saved_dict["X_scaler"] = X_scaler
        saved_dict["classifier"] = svc
        saved_dict["param"] = param

        pickle.dump(saved_dict,open("classifier_params_hardnegposmine_size.p","wb"))
        exit()
    else:
        saved_dict = pickle.load( open( "classifier_params_hardnegposmine.p", "rb" ) )
        param = saved_dict["param"]

    frame = Frame(10)

    ## Use these when running in parallel
    # search_window_sizes_d = {'0': [192,128,96],
    #                        '1': [64],
    #                        '2': [32]}
    # ylims_d = {'0': [[360,704],[376,704],[400,650]],
    #            '1': [[400,650]],
    #            '2': [[400,464]]}

    # Use these when running in serial
    search_window_sizes = [128,96,64,32]
    ylims = [[376,704],[400,650],[400,650],[400,464]]
    

    # Get individual frame from the project video
    # Process frame and write to disk
    vidcap = cv2.VideoCapture('project_video.mp4')
    success,image = vidcap.read()
    success = True
    count = 1
    while success:
        success,image = vidcap.read()
        draw_image = np.copy(image)
        drawn_image = frame.find_labelled_cars_in_frame(image,\
                                                        saved_dict,\
                                                        search_window_sizes,\
                                                        ylims,
                                                        count,
                                                        debug=False,
                                                        num_process=1)
        file = 'vid_images/frame{:04d}.jpg'.format(count)
        cv2.imwrite(file,drawn_image)
        print("File {} written to disk...".format(file))
        count += 1


    # Once the individual frames are available on disk
    # create the video from the frames
    images = []
    for i in range(1,1253):
        images.append("frame{:04d}.jpg".format(i))

    # Determine the width and height from the first image
    image_path = os.path.join('vid_images', images[0])
    frame = cv2.imread(image_path)
    height, width, channels = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter('out5.mp4', fourcc, 15, (width, height))

    for image in images:
        image_path = os.path.join('vid_images', image)
        frame = cv2.imread(image_path)
        out.write(frame) # Write out frame to video

    out.release()
    exit()