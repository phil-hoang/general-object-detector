# ==================================================================== #
#                       Distance approximation                         #
# ==================================================================== #

import cv2 as cv
import numpy as np

import visualizer.coco as coco
import visualizer.pascal as pascal 
from utils2 import constants as constants


def get_focal_length(video_file):
    """
    Focal lengths for different video sources.
    """
    camera1 = 700
    camera2 = 800

    focal_lengths = {"freeway-1": camera1, "freeway-2": camera1, 
                    "freeway-3": camera1, "suburban-1": camera2, 
                    "suburban-2": camera2, "suburban-3": camera2,
                    "suburban-4": camera2, "suburban-5": camera2, 
                    "suburban-6": camera2, "urban-1": camera1, 
                    "urban-2": camera1, "urban-3": camera1, 
                    "urban-4": camera1, "urban-5": camera2, 
                    "urban-6": camera2, "urban-7": camera2}
    
    try:
        fl = focal_lengths[video_file]

    except KeyError:
        print("No valid focal length found for this video.\nTo use the " 
         + "distance feature please add one in utils2/distance.\nExiting...")
        exit()
    
    return fl


def vehicles(frame, boxes, count, focal_length):
    """
    Vehicle class specific distance approximation based on focal length.
    Rounds distance to the nearest 5m.
    Parameters to tune:
    - vehicle_width
    - Aspect ratio height to width

    Return:
    Frame with overlaid distances
    """
    
    box = boxes[count].numpy()
    width = None

    # Get box dimensions.
    x1, x2 = box[0], box[2]
    y1, y2 = box[1], box[3]
    width = x2 - x1
    height_to_width = (y2 - y1) / (x2 - x1)
       
    # Compute ROI for boxes. We only care about the centre half of the frame.
    xmin = frame.shape[1] / 4
    xmax = frame.shape[1] * 3 / 4 
    
    if x1 < xmin or x2 > xmax:
        return frame
              
    # Parameter which work well. Assumption is a view from the rear.
    vehicle_width = 2.0 # width [m]

    # Filter aspect ratio to prevent computing on vehicle side views. Round result to 5m.
    if (width != None) and (height_to_width > .5):
        distance = np.around((vehicle_width * focal_length) / (width*5), decimals=0)*5
        bb_text = "{:.0f}m".format(distance)
        font = cv.FONT_HERSHEY_SIMPLEX
        cv.putText(frame, bb_text, (x1,int(y1-5)), font, 0.8, constants.box_colours()["motor"], 2)

    return frame


def pedestrians(frame, boxes, count, focal_length):
    """
    Pedestrian class specific distance approximation based on focal length.
    Rounds distance to the nearest 5m.
    Parameters to tune:
    - pedestrian_width
    - Aspect ratio height to width

    Return:
    Frame with overlaid distances
    """
    
    box = boxes[count]
    width = None

    # Get box dimensions
    x1, x2 = box[0], box[2]
    y1, y2 = box[1], box[3]
    width = x2 - x1
    height_to_width = (y2 - y1) / (x2 - x1)

    # Compute ROI for boxes. We only care about the centre half of the frame.
    xmin = frame.shape[1] / 4
    xmax = frame.shape[1] * 3 / 4 
    
    if x1 < xmin or x2 > xmax:
        return frame
  
    # Parameter which work well. Assumption is a view from the rear.
    pedestrian_width = .65 # width [m]

    # Filter aspect ratio to prevent computing on vehicle side views. Round result to 5m.
    if (width != None) and (height_to_width > 1.8):
        distance = np.around((pedestrian_width * focal_length) / (width*5), decimals=0)*5
        
        bb_text = "{:.0f}m".format(distance)
        font = cv.FONT_HERSHEY_SIMPLEX
        cv.putText(frame, bb_text, (x1,int(y1-5)), font, 0.8, constants.box_colours()["person"], 2)

    return frame


def stop_signs(frame, boxes, count, focal_length):
    """
    Stop sign class specific distance approximation based on focal length.
    Rounds distance to the nearest 5m.
    Parameters to tune:
    - stopsign_width (there are two sizes, we took something closer to the smaller size)
    - Aspect ratio height to width

    Return:
    Frame with distance displayed next to the symbol.
    """
    
    box = boxes[count]
    width = None

    # Get box dimensions
    x1, x2 = box[0], box[2]
    y1, y2 = box[1], box[3]
    width = x2 - x1
    height_to_width = (y2 - y1) / (x2 - x1)

    # Compute ROI for boxes.
    xmin = frame.shape[1] / 8
    xmax = frame.shape[1] * 7 / 8 
    
    if x1 < xmin or x2 > xmax:
        return frame
               
    # Parameter which work well. Assumption is a view from the rear.
    stopsign_width = .8 # width [m]

    # Filter aspect ratio to prevent computing on vehicle side views. Round result to 5m.
    if (width != None) and (height_to_width > .7):
        distance = np.around((stopsign_width * focal_length) / (width*5), decimals=0)*5
        bb_text = "{:.0f}m".format(distance)
        font = cv.FONT_HERSHEY_SIMPLEX

        # Calculate top centre position to place stop sign.
        y_offset = 80
        x_offset = 70
       
        x1 = int((frame.shape[1]/2) + x_offset)
        y1 = y_offset

        cv.putText(frame, bb_text, (x1,int(y1)), font, 0.8, (0, 0, 255), 2)

    return frame



def estimate(frame, boxes, model_type, labels, video_file):
    """
    Estimates distance for each box for selected classes vehicles and pedestrian.
    Uses fixed approximations for focal width [pixels] and object width.
    """
    
    # Focal length in pixels.
    focal_length = get_focal_length(video_file)

    if (model_type in coco.supported_models()):
        count = 0

        for label in labels:
            if label in [coco.labels()["car"], coco.labels()["truck"], coco.labels()["bus"]]:
                frame = vehicles(frame, boxes, count, focal_length)

            elif label in [coco.labels()["person"]]:
                frame = pedestrians(frame, boxes, count, focal_length)

            elif label in [coco.labels()["stopsign"]]:
                frame = stop_signs(frame, boxes, count, focal_length)

            count += 1

    elif (model_type in pascal.supported_models()):
        count = 0

        for label in labels:       
            if label in [pascal.labels()["car"], coco.labels()["bus"]]:
                frame = vehicles(frame, boxes, count, focal_length)

            elif label in [pascal.labels()["person"]]:
                frame = pedestrians(frame, boxes, count, focal_length)

            count += 1
    
    return frame