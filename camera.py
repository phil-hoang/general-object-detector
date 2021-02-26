"""
Main code to use different models with a webcam.

Currently supported models and arguments to call it:
SSD with Mobilenet          | -ssdm
SSD with Mobilenet Lite     | -ssdmlite


The ssd model is from: https://github.com/qfgaohao/pytorch-ssd
"""

import numpy as np
import cv2 as cv
import time
import sys
import torch

from ssd_pytorch.ssd import ssdModel as ssd
from visualizer.pascal import drawBoxes as pascalBoxes
from visualizer.stats_core import showStats as showCoreStats
from visualizer.stats_model import showStats as showModelStats


# Required for the slider
def nothing(x):
    pass

#%%
def runProgram():
    
    #%% Model selection if chosen in command line
    

    if ( (len(sys.argv) == 2) and (model_type == "-ssdm")):
        net, predictor = ssd("-ssdm")
    elif ( (len(sys.argv) == 2) and (model_type == "-ssdmlite")):
        net, predictor = ssd("-ssdmlite")
    elif ( (len(sys.argv)==2)) and (model_type == "-detr"):
        model = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=True)
    else:
        model_enabled = 0
    
    # Prepare camera
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR! Cannot open camera")
        exit()

    # Create slider to turn stats and model on or off
    cv.namedWindow('Live Detection')
    switch = 'Model OFF / ON'
    cv.createTrackbar('Show stats', 'Live Detection', 0, 1, nothing)
    if (len(sys.argv) == 2):
        cv.createTrackbar(switch, 'Live Detection', 0, 1, nothing)

    # Initialize list for model unrelated core stats. [fps, time.start, time.end]
    stats_core = [None, None, None]

    #%% Loop through each frame
    while True:
        # Get time before detection
        stats_core[1] = time.time()
        
        # Get a frame, convert to RGB and get frames per second fps
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        stats_core[0] = cap.get(cv.CAP_PROP_FPS)

        # Set slider to turn on or off stats and enable or disable a model, if a model is selected
        statsFlag = cv.getTrackbarPos('Show stats','Live Detection')
        if (len(sys.argv) == 2):
            model_enabled = cv.getTrackbarPos(switch,'Live Detection')
        
        # Locate objects with model if selected
        if (len(sys.argv) == 2 and model_enabled == 1 and model_type == "-detr"):
            t_image = torch.as_tensor(image, dtype=torch.float32).unsqueeze(0)
            t_image = t_image.permute(0, 3, 1, 2)
            output = model(t_image)
            # output is a dict containing "pred_logits" of [batch_size x num_queries x (num_classes + 1)]
            # and "pred_boxes" of shape (center_x, center_y, height, width) normalized to be between [0, 1]

        elif (len(sys.argv) == 2 and model_enabled == 1):
            boxes, labels, probs = predictor.predict(image, 10, 0.4)
            frame = pascalBoxes(image, probs, boxes, labels)
        
        # Get time after detection
        stats_core[2] = time.time()

        #  Display stats if selected with slider
        if (statsFlag == 1):
            frame = showCoreStats(frame, stats_core) 
        if (statsFlag == 1) and (model_enabled == 1):
            frame = showModelStats(frame, labels)

        # Display the resulting frame
        cv.imshow('Live Detection', frame)
        if cv.waitKey(1) == ord('q'):
            break

    # When everything is done, release the capture
    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    # Allow no model or selected model
    if (len(sys.argv) == 1 ):
        model_type = None
    elif (len(sys.argv) == 2 and (sys.argv[1] == "-ssdm")):
        model_type = "-ssdm"
    elif (len(sys.argv) == 2 and (sys.argv[1] == "-ssdmlite")):
        model_type = "-ssdmlite"
    elif (len(sys.argv) == 2 and (sys.argv[1] == "-detr")):
        model_type = "-detr"
    else:
        print("Usage: no arg or -ssdm or -ssdmlite")
        exit()
        
    print("Starting camera ... \nPress q to exit ")
    runProgram()