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

    # Label/class names in COCO
    instance_labels = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]


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