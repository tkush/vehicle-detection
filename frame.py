from object_functions import *
from multiprocessing import Process, Queue
import os
from copy import deepcopy

# This the function called by all worker processes
def worker(idx, frame, draw_on_img, dict_of_params, window, y_lim, bbox_list):
    bbox = Get_Features.find_cars_my(frame,\
                                     draw_on_img,\
                                     dict_of_params,\
                                     window,\
                                     y_lim)
    bbox_list.put(bbox)

# Vehicle class
class Vehicle():

    def __init__(self, label, num=5):
        self._label = label
        self._cur_centroid = None
        self._last_centroid = None
        self._cur_bbox = None
        self._last_n_bbox = []
        self._detected = False
        self._num = num
    
# Frame class
class Frame():
    
    def __init__(self, num_frames_to_track=5):
        self._is_veh_detected = False
        self._last_frame_veh = []
        self._cur_frame_vehicles = []
        self._true_positives = []
        self._last_n_frames_vehicles = []
        self._track_n_frames = num_frames_to_track
        self._tracked_vehicles = {}

    # Function to calculate overlapping % b/w two bboxes in consective frames
    def get_overlap_area_ratio(self, bbox1, bbox2):
        x1_min = bbox1[0][0]
        x1_max = bbox1[1][0]
        y1_min = bbox1[0][1]
        y1_max = bbox1[1][1]
        x2_min = bbox2[0][0]
        x2_max = bbox2[1][0]
        y2_min = bbox2[0][1]
        y2_max = bbox2[1][1]

        area_1 = ( x1_max - x1_min ) * ( y1_max - y1_min )
        area_2 = ( x2_max - x2_min ) * ( y2_max - y2_min )
        dx = min( x1_max, x2_max) - max( x1_min, x2_min )
        dy = min( y1_max, y2_max) - max( y1_min, y2_min )
        if ( dx >=0 and dy >=0 ):
            area_inter = dx * dy
        else:
            area_inter = 0
        area_union = area_1 + area_2 - area_inter
        return float( area_inter / area_union )

    # Function to check validity of detections and remove
    # false positives
    def check_validity(self, tolerance):
        if ( len(self._last_frame_veh) == 0 ):
            # Nothing in the previous frames, just add whatever is found
            self._last_frame_veh = self._cur_frame_vehicles
            self._is_veh_detected = False
            # print("Last list empty. Added {} vehicle detections".format(self._cur_frame_vehicles))
        else:
            # Check incoming detections against the last one
            temp = []
            for vehicle in self._cur_frame_vehicles:
                overlap_area = []
                for found_veh in self._last_frame_veh:
                    overlap_area.append( self.get_overlap_area_ratio( vehicle[1],found_veh[1] ) )
                    print("Comparing...")
                    print("\t{}".format(vehicle))
                    print("With")
                    print("\t{}".format(found_veh))
                    # print("\nDistance = {}".format(dist[-1]))
                    print("Overlap area ratio = {}".format(overlap_area[-1]))
                idx_min = np.argmax(overlap_area)
                
                # print("Vehicle dist {}".format(dist[idx_min]))
                if ( overlap_area[idx_min] >= 0.2 ):#dist[idx_min] <= tolerance or  ):
                    # found a match
                    temp.append(vehicle)
                    print("Found valid vehicle")
            if ( len( temp ) > 0 ):
                self._true_positives = temp
                self._last_frame_veh = self._cur_frame_vehicles
                self._is_veh_detected = True
            else:                
                print("No valid vehicle found")
                self._true_positives = []
                self._is_veh_detected = False
                self._last_frame_veh = self._cur_frame_vehicles


    # Copy detections to class member and check for validity
    def check_detections(self, detected, tolerance):
        self._cur_frame_vehicles = detected
        self.check_validity(tolerance)

    # Merge bboxes together based on tol
    def merge_bboxes(self, pos, neg, tolerance):
        merged_boxes = []
        for veh_n in neg:
            dist = []
            neg_c_x = veh_n[2][0]
            neg_c_y = veh_n[2][1]
            for veh_p in pos:
                pos_c_x = veh_p[2][0]
                pos_c_y = veh_p[2][1]
                dist.append( np.sqrt( ( neg_c_x - pos_c_x )**2 + \
                                      ( neg_c_y - pos_c_y )**2 ) )
                # print("Comparing...")
                # print("\t{}".format(veh_n))
                # print("With")
                # print("\t{}".format(veh_p))
                # print("\nDistance = {}".format(dist[-1]))
            if ( len(dist) > 0) :
                idx_min = np.argmin(dist)
                if ( dist[idx_min] <= tolerance ):
                    # found a match
                    veh_p = pos[idx_min]
                    x_coords = [veh_n[1][0][0], veh_n[1][1][0], veh_p[1][0][0], veh_p[1][1][0]]
                    y_coords = [veh_n[1][0][1], veh_n[1][1][1], veh_p[1][0][1], veh_p[1][1][1]]
                    x_min = min(x_coords)
                    x_max = max(x_coords)
                    y_min = min(y_coords)
                    y_max = max(y_coords)
                    centroid = (int((x_min + x_max)/2),\
                                int((y_min + y_min)/2))
                    merged_boxes.append([veh_p[0],( (x_min,y_min), (x_max,y_max) ), centroid])
                    del pos[idx_min]
        
        if ( len (merged_boxes) > 0 ):
            for veh in pos:
                merged_boxes.append(veh)
            return merged_boxes
        else:
            return pos

    # Main function called to detect vehicles in image frame        
    def find_labelled_cars_in_frame(self, frame, dict_of_params, window_sizes, y_lims, count, debug=False, num_process=4):
        colors = [(0,0,255),\
                  (0,255,0),\
                  (255,0,0),\
                  (255,0,255),\
                  (0,255,255),\
                  (255,255,0),\
                  (0,0,0),\
                  (255,255,255)]
        draw_on_img = frame.copy()
        master_list_bbox = []
            
        if ( num_process > 1 ):
            # Put output in the queue
            bbox_q = Queue()
            process_pool = []
            # num_windows = len(window_sizes)
            # win_per_proc = num_windows//num_process
            for i in range(num_process):
                # print("Process {0}, running on windows {1}".format(i,window_sizes[str(i)]))
                process = Process(target = worker, 
                                args   = (i,\
                                        frame,\
                                        draw_on_img,\
                                        dict_of_params,\
                                        window_sizes[str(i)],\
                                        y_lims[str(i)],\
                                        bbox_q)
                                        )
                process_pool.append(process)
                process.start()
            
            for i in range(num_process):
                master_list_bbox.append(bbox_q.get())
            
            for p in process_pool:
                p.join()

            flattened_master_list = [item for nestlist in master_list_bbox for item in nestlist]
        
        else:
            flattened_master_list = Get_Features.find_cars_my(frame,\
                                            draw_on_img,\
                                            dict_of_params,\
                                            window_sizes,\
                                            y_lims)
        
        heat_img = np.zeros_like(frame[:,:,0]).astype(np.float)        
        heatmap = Get_Features.add_heat(heat_img, flattened_master_list)
        thresholded = Get_Features.apply_threshold(heatmap,2)
        thresholded = np.clip(thresholded, 0, 255)
        labels = label(thresholded)
        
        if ( labels[1] > 0 ):
            temp_detections = []
            false_det = []
            # something except background is present
            # os.system('cls')                    
            for car_number in range( 1, labels[1]+1 ):
                # Find pixels with each car_number label value
                nonzero = (labels[0] == car_number).nonzero()
                # Identify x and y values of those pixels
                nonzeroy = np.array(nonzero[0])
                nonzerox = np.array(nonzero[1])
                # Define a bounding box based on min/max x and y
                bbox = ((np.min(nonzerox), np.min(nonzeroy)),\
                        (np.max(nonzerox), np.max(nonzeroy)))
                # Check #1: The bbox must be minimum of 30x30 - this eliminates small 
                # detections
                height = bbox[1][1] - bbox[0][1]
                width = bbox[1][0] - bbox[0][0]
                centroid = (int(bbox[0][0] + bbox[1][0])/2,\
                            int(bbox[0][1] + bbox[1][1])/2)
                if ( height >= 30 and width >=30 ):  
                    temp_detections.append([count,bbox,centroid])
                else:
                    false_det.append([count,bbox,centroid])
                
                if ( debug ):
                    print("\n")
                    print("******Frame # {}**********".format(count))
                    print("Car @ {}".format(bbox))
                    print("Car cetroid {}".format(centroid))

            merged_boxes = self.merge_bboxes(temp_detections, false_det, 100)
            self.check_detections( merged_boxes, 30 )                  
                    
            # Draw the box on the image
            if ( self._is_veh_detected ):
                for idx, detection in enumerate(self._true_positives):
                    bbox = detection[1]
                    cv2.rectangle(draw_on_img, bbox[0], bbox[1], colors[idx], 2)
                            
        return draw_on_img
