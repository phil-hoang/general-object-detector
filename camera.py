"""
Main code to use different models with a webcam.

Currently supported models and arguments to call it:
SSD with Mobilenet          | -ssdm
SSD with Mobilenet Lite     | -ssdmlite
DETR                        | -detr


The ssd model is from: https://github.com/qfgaohao/pytorch-ssd
"""

import numpy as np
import cv2 as cv
import time
import sys
import torch

from ssd_pytorch.ssd import ssdModel as ssd
from faster_rcnn.fasterrcnn import fasterRcnnModel as frcnn
from faster_rcnn.fasterrcnn import predict
from visualizer.pascal import drawBoxes as pascalBoxes
from visualizer.stats_core import showStats as showCoreStats
from visualizer.stats_model import showStats as showModelStats
import visualizer.signs as signs
import torchvision.transforms as T
from visualizer.coco import draw_boxes as cocoBoxes

transform = T.Compose([
    T.ToPILImage(),
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b


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
    # DETR requires pytorch version 1.5+ and torchvision 0.6+
    elif ( (len(sys.argv)==2)) and (model_type == "-detr"):
        model = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=True)
        model.eval()
    elif ( (len(sys.argv) == 2) and (model_type == "-fasterrcnn")):
            predictor = frcnn()
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

    # Load sign symbols
    stop_sign = signs.load()[0]

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
            t_image = transform(image).unsqueeze(0)
            output = model(t_image)

            # keep only predictions of 0.9+ confidence
            probas = output['pred_logits'].softmax(-1)[0,:,:-1]
            #keep = probas.max(-1).values >= 0.5

            boxes = rescale_bboxes(output['pred_boxes'][0], (t_image.size()[3], t_image.size()[2])).detach()
            #boxes[:, [2,3]] = boxes[:, [3,2]]
            #probs = probas[keep]
            
            #labels = [CLASSES[x]  for x, i in zip(probas.max(-1).indices, keep) if (i == True) ]
            labels = probas.max(-1).indices


            predictions = { 'boxes' : boxes,
                            'scores' : probas.max(-1).values,
                            'labels' : labels}

            frame = cocoBoxes(image, predictions)   # Until the function above is implemented
            # output is a dict containing "pred_logits" of [batch_size x num_queries x (num_classes + 1)]
            # and "pred_boxes" of shape (center_x, center_y, height, width) normalized to be between [0, 1]
        elif (len(sys.argv) == 2 and model_enabled == 1):
            boxes, labels, probs = predictor.predict(image, 10, 0.4)
            frame = pascalBoxes(image, probs, boxes, labels)
        #if (len(sys.argv) == 2 and model_enabled == 1):
        #    boxes, labels, probs = predictor.predict(image, 10, 0.4)
        #    frame = pascalBoxes(image, probs, boxes, labels)

        ###### FASTER RCNN TEST
        #pred = predict(predictor, image, 10, 1)
        #labels = [1, 1]
        #probs = [1,1]
        
        # Get time after detection
        stats_core[2] = time.time()

        #  Display stats if selected with slider
        if (statsFlag == 1):
            frame = showCoreStats(frame, stats_core) 
        if (statsFlag == 1) and (model_enabled == 1):
            frame = showModelStats(frame, labels)

        # Enable symbols
        if (model_enabled == 1):
            #frame = signs.showStopSign(frame, stop_sign, labels, probs)
            pass

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
    elif (len(sys.argv) == 2 and (sys.argv[1] == "-fasterrcnn")):
        model_type = "-fasterrcnn"
    else:
        print("Usage: no arg or -ssdm or -ssdmlite or -fasterrcnn or -detr")
        exit()
        
    print("Starting camera ... \nPress q to exit ")
    runProgram()