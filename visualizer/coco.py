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
    img         --
    predictions -- Dictionary with boxes, labels and scores. Not a list of dict!
    
    Returns:
    img_out     -- Frame with bounding boxes
    """

    # Label names
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

    # Colours
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

    ### Make boxes
    scores = predictions['scores'].detach().numpy()
    scores = scores[scores > thresh]

    # Boxes
    boxes = list(predictions['boxes'].detach().numpy())
    labels_id = list(predictions['labels'].detach().numpy())


    img_out = img

    # Iterate through each instance
    for obj in range(len(scores)):

        box = boxes[obj]

        # Calculate coordinates for bounding box
        x1 = box[0]
        x2 = box[2]
        y1 = box[1]
        y2 = box[3]

        label = instance_labels[labels_id[obj]]
        print(label)
        print(scores[obj])
        font = cv.FONT_HERSHEY_PLAIN
        img_out = cv.rectangle(img,(x1,y1),(x2,y2),colour[obj],2)

        # Add label to bounding box
        bb_text = label + " " + "{:.2f}".format(scores[obj])
        cv.putText(img_out, bb_text, (x1,int(y1-5)), font, 1.3, (0,0,0), 2 )
        
    return img_out