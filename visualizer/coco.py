import numpy as np
import cv2 as cv


def draw_boxes(image, boxes, labels, conf, thresh=0.9):
    """
    Draws boxes per frame for COCO data.

    Args:
    imgage      -- Original image without bounding boxes
    boxes       -- List of coordinates of the top left and bottom right of the bounding box ordered as [(x1, y1, x2, y2)]
    labels      -- List of index labels for each bounding box [<label indices>]
    scores      -- List of class confidence scores for each bounding box [<class scores>]. For COCO, expects 91 different classes.
    
    Returns:
    img_out     -- image now with bounding boxes with labels and scores top left of the box
    """

    # Class label indices
    labelsMotor = [6, 3, 4, 8]
    labelsPerson = [1, 2]
    labelsSigns = [10, 13]

    # Colours used for the bounding boxes
    colourMotor = (255, 0, 0)
    colourPerson = (0, 0, 255)
    colourSigns = (0, 255, 0)
    colourOther = (255, 165, 0)

    # TODO: rename
    labels_id = labels
    # Iterate through each instance
    for i in range(len(conf)):
         # Filter for classes
        if (labels_id[i] in labelsMotor):
            box = boxes[i, :]
            cv.rectangle(image, (box[0], box[1]), (box[2], box[3]), colourPerson, 2)
        elif (labels_id[i] in labelsPerson):
            box = boxes[i, :]
            cv.rectangle(image, (box[0], box[1]), (box[2], box[3]), colourMotor, 2)
        elif (labels_id[i] in labelsSigns):
            box = boxes[i, :]
            cv.rectangle(image, (box[0], box[1]), (box[2], box[3]), colourSigns, 2)
        else:
            box = boxes[i, :]
            cv.rectangle(image, (box[0], box[1]), (box[2], box[3]), colourOther, 2)

    image = cv.cvtColor(image, cv.COLOR_RGB2BGR)   
    return image