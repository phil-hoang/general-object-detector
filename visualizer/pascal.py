import cv2 as cv
from utils2 import constants

def draw_boxes(image, probs, boxes, labels):
    """
    Draws bounding boxes for selected classes of the Pascal dataset.
   
    Box colours are:
    Motorized vehicles  | Blue
    Bicycles, person    | Red
    
    Args:
    image       - Opencv image object in RGB
    probs       - Confidences for the detections
    boxes       - Coordinates of the bounding boxes
    labels      - Torch tensor with label indices

    Returns:
    image       - Opencv image in BGR with bounding boxes
    """
    labels_motor = [6, 7, 14]
    labels_person = [2, 15]

    colour_motor = constants.box_colours()["motor"]
    colour_person = constants.box_colours()["person"]
    colour_other = constants.box_colours()["other"]

    for i in range(boxes.size(0)):
                # Filter for classes
                if (labels[i] in labels_motor):
                    box = boxes[i, :]
                    cv.rectangle(image, (box[0], box[1]), (box[2], box[3]), colour_person, 2)
                elif (labels[i] in labels_person):
                    box = boxes[i, :]
                    cv.rectangle(image, (box[0], box[1]), (box[2], box[3]), colour_motor, 2)
                else:
                    box = boxes[i, :]
                    cv.rectangle(image, (box[0], box[1]), (box[2], box[3]), colour_other, 2)
    
    # Convert colour channels back to BGR
    image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
    return image


def supported_models():
    """
    Returns a list with the currently supported models.
    """
    models = ["ssdm", "ssdmlite"]

    return models


def labels():
    """
    Returns a dict with relevant labels as keys and their index as value.
    """
    labels = {"car": 7, "bus": 6, "motorcycle": 14, "bicycle": 2, "person": 15}

    return labels