import cv2 as cv
import numpy as np

import visualizer.coco as coco
import visualizer.pascal as pascal 
from utils2 import constants as constants
 
"""
TODO:
* Filter boxes for vehicle class and pedestrian class only
* Add stop sign and add distance info above next to sign symbol

* Tune parameters
* Think about how distance warning could work without unknown vehicle speed
    -> Check for approach by checking rate of decrease of distance over some frames

* Write post about it
"""

def get_focal_length():
    focal_length = {"video": 1000}
    return focal_length


def vehicles(frame, boxes, count):
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
     
    # Focal length in pixels.
    focallength = get_focal_length()["video"]
         
    # Parameter which work well. Assumption is a view from the rear.
    vehicleWidth = 2.0 # width [m]

    # Filter aspect ratio to prevent computing on vehicle side views. Round result to 5m
    if (width != None) and (height_to_width > .7):
        distance = np.around((vehicleWidth * focallength) / (width*5), decimals=0)*5
        bb_text = "{:.0f}m".format(distance)
        font = cv.FONT_HERSHEY_SIMPLEX
        cv.putText(frame, bb_text, (x1,int(y1-5)), font, 0.8, constants.box_colours()["motor"], 2)

    return frame


def pedestrians(frame, boxes, count):
    box = boxes[count]
    width = None

    # Get box dimensions
    x1, x2 = box[0], box[2]
    y1, y2 = box[1], box[3]
    width = x2 - x1
    height_to_width = (y2 - y1) / (x2 - x1)
           
    # Focal length in pixels.
    focallength = get_focal_length()["video"]
         
    # Parameter which work well. Assumption is a view from the rear.
    pedestrianWidth = .6 # width [m]

    # Filter aspect ratio to prevent computing on vehicle side views. Round result to 5m
    if (width != None) and (height_to_width > 2):
        distance = np.round((pedestrianWidth * focallength) / (width))
        
        bb_text = "{:.0f}m".format(distance)
        font = cv.FONT_HERSHEY_SIMPLEX
        cv.putText(frame, bb_text, (x1,int(y1-5)), font, 0.8, constants.box_colours()["person"], 2)

    return frame



def estimate(frame, boxes, model_type, labels):
    """
    Estimates distance for each box for selected classes vehicles and pedestrian.
    Uses fixed approximations for focal width [pixels] and vehicle width.
    """

    if (model_type in coco.supported_models()):
        count = 0

        for label in labels:
            if label in [coco.labels()["car"], coco.labels()["truck"], coco.labels()["bus"]]:
                frame = vehicles(frame, boxes, count)

            elif label in [coco.labels()["person"]]:
                frame = pedestrians(frame, boxes, count)

            count += 1

    elif (model_type in pascal.supported_models()):
        count = 0

        for label in labels:       
            if label in [pascal.labels()["car"], coco.labels()["bus"]]:
                frame = vehicles(frame, boxes, count)

            elif label in [pascal.labels()["person"]]:
                frame = pedestrians(frame, boxes, count)

            count += 1

    return frame