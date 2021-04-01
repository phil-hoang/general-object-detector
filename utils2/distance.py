import cv2 as cv
import numpy as np
 
"""
TODO:
* Filter boxes for vehicle class only
* Make region of interest to filter location where distance is computed
* Tune parameters
* Think about how distance warning could work without known vehicle speed

* Write post about it
"""


def estimateDistance(boxWidth):
    pass
    
    return 



def estimate(frame, boxes):
    """
    Estimates distance for each box.
    Uses fixed approximations for focal width [pixels] and vehicle width.
    """
    
    #print("---")
    for box in boxes:
            width = None
            if len(box) > 0:
                x1, x2 = box[0], box[2]
                y1, y2 = box[1], box[3]
                width = x2 - x1
                height_to_width = (y2 - y1) / (x2 - x1)

                # Focal length in pixels
                focallength = 1000
                vehicleWidth = 2.0
                if (width != None) and (height_to_width > .8):
                    distance = np.round((vehicleWidth * focallength) / width)
                    # print("Distance: {:.2f}m".format(distance))
                    bb_text = "{:.0f}m".format(distance)
                    font = cv.FONT_HERSHEY_SIMPLEX

                    cv.putText(frame, bb_text, (x1,int(y1-5)), font, 0.8, (255, 0, 0), 2)


    return frame