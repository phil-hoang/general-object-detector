"""
Main code to use different models with a webcam or a video file.
To use the detector with for example SSD Mobilenet on file video.mp4, type:

    run.py -ssdm video

To use it with the webcam just ommit the filename:

    run.py -ssdm

Running the command

    run.py

without any arguments just opens the webcam and displays its output.


Currently supported models and arguments to call it:
SSD with Mobilenet          | -ssdm
SSD with Mobilenet Lite     | -ssdmlite
SSD with VGG-16             | -ssdvgg       -> TODO
YOLO v?                     | -yolo         -> TODO
DETR with Resnet50          | -detr         -> TODO
Faster R-CNN with ?         | -fasterrcnn   -> TODO



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


# Required for the slider
def nothing(x):
    pass

#%%
def runProgram(model_type, video_file):
    #%% Model selection if chosen in command line
    if ((model_type == "-ssdm") or (model_type == "-ssdmlite")):
        net, predictor = ssd(model_type)
    elif (model_type == "-fasterrcnn"):
        predictor = frcnn()
    elif (model_type == "-detr"):
        print("DETR")
    else:
        model_enabled = 0

    # Prepare input and output
    if (video_file == None):
        # Camera mode
        cap = cv.VideoCapture(0)
        if not cap.isOpened():
            print("ERROR! Cannot open camera")
            exit()
    else:
        # Video mode
        cap = cv.VideoCapture("media/DrivingClips/" + video_file + ".mp4")
        #out = cv.VideoWriter('dev/output.avi', cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), 12.0, (1280, 720))
        if not cap.isOpened():
            print("ERROR! Cannot read video")
            exit()
    
    # Create slider to turn stats and model on or off
    statsSliderLabel = 'Show stats'
    modelSliderLabel = 'Model OFF / ON'
    if (len(sys.argv) <= 2):
        windowname = 'Live Detection'
    else:
        windowname = 'Video Detection'
    cv.namedWindow(windowname)
    cv.createTrackbar(statsSliderLabel, windowname, 1, 1, nothing)
    if (len(sys.argv) >= 2):
        cv.createTrackbar(modelSliderLabel, windowname, 1, 1, nothing)
    
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
        statsFlag = cv.getTrackbarPos(statsSliderLabel, windowname)
        if (len(sys.argv) >= 2):
            model_enabled = cv.getTrackbarPos(modelSliderLabel, windowname)

        # Locate objects with model if selected
        if (len(sys.argv) >= 2 and model_enabled == 1):
            boxes, labels, conf = predictor.predict(image, 10, 0.4)
            frame = pascalBoxes(image, conf, boxes, labels)

         # Get time after detection
        stats_core[2] = time.time()

        #  Display stats if selected with slider
        if (statsFlag == 1):
            frame = showCoreStats(frame, stats_core) 
        if (statsFlag == 1) and (model_enabled == 1):
            frame = showModelStats(frame, labels, conf)

        # Enable symbols
        if (model_enabled == 1):
            frame = signs.showStopSign(frame, stop_sign, labels, conf)
        
        # Display the resulting frame
        cv.imshow(windowname, frame)
        #out.write(frame)
        if cv.waitKey(1) == ord('q'):
            break

    # When everything is done, release the capture
    cap.release()
    #out.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    # Allow no model or selected model
    supported_models = ["-ssdm", "-ssdmlite", "-ssdvgg", "-detr", "-fasterrcnn"]
    
    # Camera mode without a model
    if (len(sys.argv) == 1 ):
        model_type = None
        video_file = None
    # Camera mode with a model
    elif (len(sys.argv) == 2 and (sys.argv[1] in supported_models)):
        model_type = sys.argv[1]
        video_file = None
    # Camera mode with a video file
    elif (len(sys.argv) == 3 and (sys.argv[1] in supported_models)):
        model_type = sys.argv[1]
        video_file = sys.argv[2]
    else:
        print("Usage: <model> <video_filename>\nAvailable models are: -ssdm, -ssdmlite, -ssdvgg, -fasterrcnn, -detr\nTo just run the webcam provide no args.")
        exit()

    if (len(sys.argv) <= 2):
        print("Starting camera ... \nPress q to exit ")
    else:
        print("Starting video ... \nPress q to exit ")
    runProgram(model_type, video_file)