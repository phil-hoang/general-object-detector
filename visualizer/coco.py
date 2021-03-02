import numpy as np
import cv2 as cv

#import torch
#import torchvision
#from torchvision.transforms import ToTensor


#%%


def draw_boxes(img, predictions, thresh=0.9):
    """
    Per frame

    Args:
    img         -- Original image without bounding boxes
    predictions -- Dictionary with boxes, labels and scores. Not a list of dict! Sorted for each bounding box
        boxes   -- Coordinates of the top left and bottom right of the bounding box ordered as [:, (x1, y1, x2, y2)]
        labels  -- index labels for each bounding box [:, <label index>]
        scores  -- Class confidence scores for each bounding box [:, <class scores>]. For COCO, expects 91 different classes.
    
    Returns:
    img_out     -- imgage now with bounding boxes with labels and scores top left of the box
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

    # Colours used for the bounding boxes
    colour = [(220,20,60), (0,255,0), (0,0,238), (255,215,0), (238,118,33), (238,118,33), (238,118,33), (238,118,33)
                , (238,118,33), (238,118,33), (238,118,33), (238,118,33), (238,118,33), (238,118,33), (238,118,33)
                , (238,118,33), (238,118,33), (238,118,33), (238,118,33), (238,118,33), (238,118,33), (238,118,33)
                , (238,118,33), (238,118,33), (238,118,33), (238,118,33), (238,118,33), (238,118,33), (238,118,33)
                , (238,118,33), (238,118,33), (238,118,33), (238,118,33), (238,118,33), (238,118,33), (238,118,33)
                , (238,118,33), (238,118,33), (238,118,33), (238,118,33), (238,118,33), (238,118,33), (238,118,33)
                , (238,118,33), (238,118,33), (238,118,33), (238,118,33), (238,118,33), (238,118,33), (238,118,33)
                , (238,118,33), (238,118,33), (238,118,33), (238,118,33), (238,118,33), (238,118,33), (238,118,33)
                , (238,118,33), (238,118,33), (238,118,33), (238,118,33), (238,118,33), (238,118,33), (238,118,33)
                , (238,118,33), (238,118,33), (238,118,33), (238,118,33), (238,118,33), (238,118,33), (238,118,33)]

    ### Threshold scores
    scores = predictions['scores'].detach().numpy()
    keep = scores > thresh

    # Filter out scores, boxes, and labels using threshold
    scores = scores[keep]
    boxes = predictions['boxes'].detach().numpy()[keep]
    labels_id = predictions['labels'].detach().numpy()[keep]


    img_out = img

    # Iterate through each instance
    for i in range(len(scores)):

        box = boxes[i]

        # Extract coordinates for bounding box
        x1 = box[0]
        x2 = box[2]
        y1 = box[1]
        y2 = box[3]

        # Extract labels and scores to label each bounding box
        label = instance_labels[labels_id[i]]
        print(label)
        print(scores[i])
        font = cv.FONT_HERSHEY_PLAIN

        # Draw bounding box
        img_out = cv.rectangle(img,(x1,y1),(x2,y2),colour[i],2)

        # Add label to bounding box
        bb_text = label + " " + "{:.2f}".format(scores[i])
        cv.putText(img_out, bb_text, (x1,int(y1-5)), font, 1.3, (0,0,0), 2 )
        
    img_out = cv.cvtColor(img_out, cv.COLOR_RGB2BGR)    
    return img_out