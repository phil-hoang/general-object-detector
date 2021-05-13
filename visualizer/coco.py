import numpy as np
import cv2 as cv

from utils2 import constants as constants


def draw_boxes(image, boxes, labels, conf, thresh=0.9):
    """
    Draws boxes per frame for COCO data.

    Args:
    imgage      -- Original image without bounding boxes
    boxes       -- List of coordinates of the top left and bottom right of the bounding box ordered as [(x1, y1, x2, y2)]
    labels      -- List of index labels for each bounding box [<label indices>]
    conf        -- List of class confidence scores for each bounding box [<class scores>]. For COCO, expects 91 different classes.
    
    Returns:
    img_out     -- image now with bounding boxes with labels and scores top left of the box
    """

    # Class label indices
    labels_motor = [6, 3, 4, 8]
    labels_person = [1]
    labels_signs = [10,13]
    labels_traffic_lights = [92,93,94]
    labels_bike = [2]

    # Colours used for the bounding boxes
    colour_motor = constants.box_colours()["motor"]
    colour_person = constants.box_colours()["person"]
    colour_bike = constants.box_colours()["bike"]
    colour_signs = constants.box_colours()["signs"]
    colour_other = constants.box_colours()["other"]
    
    # Iterate through each instance
    for i in range(len(conf)):
         # Filter for classes
        if (labels[i] in labels_motor):
            box = boxes[i, :].numpy()
            cv.rectangle(image, (box[0], box[1]), (box[2], box[3]), colour_person, 2)
        elif (labels[i] in labels_person):
            box = boxes[i, :].numpy()
            cv.rectangle(image, (box[0], box[1]), (box[2], box[3]), colour_motor, 2)
        elif (labels[i] in labels_bike):
            box = boxes[i, :].numpy()
            cv.rectangle(image, (box[0], box[1]), (box[2], box[3]), colour_bike, 2)
        elif (labels[i] in labels_signs):
            box = boxes[i, :].numpy()
            cv.rectangle(image, (box[0], box[1]), (box[2], box[3]), colour_signs, 2)
        elif (labels[i] in labels_traffic_lights):
            box = boxes[i, :].numpy()
            
            # Set colours
            colour_traffic_lights = {92:(255,0,0), 93:(0,100,0), 94:(255,255,255)}
            colour = colour_traffic_lights[int(labels[i].numpy())]
            colour_traffic_lights_background = {92:(255,0,0), 93:(0,100,0), 94:(0,0,0)}
            colour_background = colour_traffic_lights_background[int(labels[i].numpy())]

            cv.rectangle(image, (box[0], box[1]), (box[2], box[3]), colour, 2)

            # Add label to bounding box
            label = label_index_to_name(int(labels[i].numpy()))
            score = conf[i]
            bb_text = "{} ({:.2f})".format(label, score.numpy())
            font  = constants.stats_format()["font"]
            cv.rectangle(image, (int(box[0]),int(box[1])), (int(box[0]+235), int(box[1]-23)), colour_background, cv.FILLED)
            cv.putText(image, bb_text, (box[0],int(box[1]-5)), font, 0.6, (255,255,255),1)
        
        else:
            box = boxes[i, :].numpy()
            cv.rectangle(image, (box[0], box[1]), (box[2], box[3]), colour_other, 2)

    image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
    
    return image

def supported_models():
    """
    Returns a list with the currently supported models.
    """
    models = ["fasterrcnn", "detr", "yolov5s", "yolov5sTraffic"]

    return models

def label_names():
    """
    Returns a dict with relevant labels as keys and their index as value.
    """
    labels = {"car": 3, "truck": 6, "bus": 8, "motorcycle": 4, "bicycle": 2, "person": 1,
    "stopsign": 13, "stoplight": 10, "traffic light red": 92, "traffic light green": 93,
    "traffic light na": 94}

    return labels


def label_index_to_name(id):
    """
    Returns a hash table with relevant labels as keys and their index as value.
    """

    names = {3:"car", 6:"truck", 8: "bus", 4:"motorcycle", 2:"bicycle", 1:"person", 13:"stop sign",
    92:"traffic light red", 93:"traffic light green", 94:"traffic light na"}

    return names[id]