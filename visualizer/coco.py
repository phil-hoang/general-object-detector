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
    labelsPerson = [1]
    labelsSigns = [10, 13]
    labelsBike = [2]

    # Colours used for the bounding boxes
    colourMotor = (255, 0, 0)
    colourPerson = (0, 0, 255)
    colourBike = (255,165,0)
    colourSigns = (0, 255, 0)
    colourOther = (220, 220, 220)
    
    # Iterate through each instance
    for i in range(len(conf)):
         # Filter for classes
        if (labels[i] in labelsMotor):
            box = boxes[i, :]
            cv.rectangle(image, (box[0], box[1]), (box[2], box[3]), colourPerson, 2)
        elif (labels[i] in labelsPerson):
            box = boxes[i, :]
            cv.rectangle(image, (box[0], box[1]), (box[2], box[3]), colourMotor, 2)
        elif (labels[i] in labelsBike):
            box = boxes[i, :]
            cv.rectangle(image, (box[0], box[1]), (box[2], box[3]), colourBike, 2)
        elif (labels[i] in labelsSigns):
            box = boxes[i, :]
            cv.rectangle(image, (box[0], box[1]), (box[2], box[3]), colourSigns, 2)
        else:
            box = boxes[i, :]
            cv.rectangle(image, (box[0], box[1]), (box[2], box[3]), colourOther, 2)

    image = cv.cvtColor(image, cv.COLOR_RGB2BGR)   
    return image

def supportedModels():
    """
    Returns a list with the currently supported models.
    """
    models = ["-fasterrcnn", "-detr", "-yolov5s"]

    return models


def labels():
    """
    Returns a dict with relevant labels as keys and their index as value.
    """
    labels = {"car": 3, "truck": 6, "bus": 8, "motorcycle": 4, "bicycle": 2, "person": 1,
    "stopsign": 13, "stoplight": 10}

    return labels