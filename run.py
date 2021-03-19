"""
Main code to use different models with a webcam or a video file.
To use the detector with for example SSD Mobilenet on file video.mp4, type:

    run.py -ssdm video

To use it with the webcam just ommit the filename:

    run.py -ssdm

Running the command

    run.py
I 
without any arguments just opens the webcam and displays its output.


Currently supported models and arguments to call it:
SSD with Mobilenet          | -ssdm
SSD with Mobilenet Lite     | -ssdmlite
YOLO v5s                    | -yolo 
DETR with Resnet50          | -detr
Faster R-CNN with Resnet50  | -fasterrcnn


The ssd model is from: https://github.com/qfgaohao/pytorch-ssd
Yolo model is from here: https://github.com/ultralytics/yolov5
"""

import numpy as np
import cv2 as cv
import time
import sys
from pathlib import Path
import torch
from ssd_pytorch.ssd import ssdModel as ssd
from detr.detr import detr_load as detr
from detr.detr import detr_predict
from faster_rcnn.fasterrcnn import fasterRcnnModel as frcnn
from faster_rcnn.fasterrcnn import frcnn_predict
from yolo.yolo import yoloModel as yolo
from yolo.yolo import yolo_predict
from visualizer.pascal import drawBoxes as pascalBoxes
from visualizer.coco import draw_boxes as cocoBoxes
from visualizer.stats_core import showStats as showCoreStats
from visualizer.stats_model import showStats as showModelStats
import visualizer.signs as signs
import utils2.logger as logger
import utils2.constants as constants
import utils2.lane_detection as lanes


# Required for the slider
def nothing(x):
    pass

#%%
def runProgram(model_type, video_file, logs_enabled, writeOutput=False):
    # Sets which frame to process. E.g. 10 means predict on every 10th frame only, 1 is for all processing all frames.
    sampleNumber = 1 # Default: 1
 
    #%% Model selection if chosen in command line
    if ((model_type == "-ssdm") or (model_type == "-ssdmlite")):
        net, predictor = ssd(model_type)
    elif (model_type == "-fasterrcnn"):
        predictor = frcnn()
    elif (model_type == "-detr"):
        predictor = detr()
    elif (model_type == "-yolov5s"):
        predictor = yolo()
    else:
        model_enabled = 0

    # Prepare input and output
    if (video_file == None):
        # Camera mode
        cap = cv.VideoCapture(0)
        fps = cap.get(cv.CAP_PROP_FPS)
        dim = (int(cap.get(cv.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)) )
        if not cap.isOpened():
            print("ERROR! Cannot open camera")
            exit()
        outputName = "camera" + model_type
    else:
        # Video mode
        cap = cv.VideoCapture("media/DrivingClips/" + video_file + ".mp4")
        fps = cap.get(cv.CAP_PROP_FPS)
        dim = (int(cap.get(cv.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)) )
        if not cap.isOpened():
            print("ERROR! Cannot read video")
            exit()
        outputName = video_file + model_type

    if writeOutput == True:
        # Create folder if it doesn't exist
        Path("media/ModelOutputs").mkdir(parents=True, exist_ok=True)
        out = cv.VideoWriter('media/ModelOutputs/' + outputName + '.avi', cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, dim)
    
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

    # Initialize logs
    if logs_enabled == True:
        logs = logger.initialize()

    #%% Loop through each frame
    counter = 0
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

        if ((counter % sampleNumber) == 0):
            counter = 0
            # Locate objects with model if selected
            if (len(sys.argv) >= 2 and model_enabled == 1 and model_type != "-detr" and model_type != "-fasterrcnn" and model_type != "-yolov5s"):
                boxes, labels, conf = predictor.predict(image, 10, 0.4)
                frame = pascalBoxes(image, conf, boxes, labels)
            elif (len(sys.argv) >= 2 and model_enabled == 1 and model_type == "-detr"):
                boxes, labels, conf = detr_predict(predictor, image)
                frame = cocoBoxes(image, boxes, labels, conf)
            elif (len(sys.argv) >= 2 and model_enabled == 1 and model_type == "-fasterrcnn"):
                boxes, labels, conf = frcnn_predict(predictor, image)
                frame = cocoBoxes(image, boxes, labels, conf)
            elif (len(sys.argv) >= 2 and model_enabled == 1 and model_type == "-yolov5s"):
                boxes, labels, conf = yolo_predict(predictor, image)
                frame = cocoBoxes(image, boxes, labels, conf)

        

            # Get time after detection
            stats_core[2] = time.time()
            #  Display stats if selected with slider
            if (statsFlag == 1):
                frame = showCoreStats(frame, stats_core) 
            if (statsFlag == 1) and (model_enabled == 1):
                frame, model_stats = showModelStats(frame, model_type, labels, conf)

            # Enable symbols
            if (model_enabled == 1):
                frame = signs.showStopSign(frame, model_type, stop_sign, labels, conf)
            
            # Write logs if enables
            if logs_enabled is True:
                logs = logger.writeLog(logs, stats_core[1], stats_core[2], labels, conf, model_stats)

            # Lane detection
            #frame = lanes.detect(frame)

            # Display the resulting frame
            cv.imshow(windowname, frame)
            if ((writeOutput == True)):
                out.write(frame)

        counter = counter + 1
        if cv.waitKey(1) == ord('q'):
            break
    
    # Writing logs to file
    if logs_enabled is True:
        logger.saveLogs(logs, video_file, model_type)

    # When everything is done, release the capture
    cap.release()
    if ((writeOutput == True) and (video_file != None)):
        out.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    # Allow no model or selected model
    supported_models = ["-ssdm", "-ssdmlite", "-ssdvgg", "-detr", "-fasterrcnn", "-yolov5s"]
    
    # Initialize log and write output variables
    logs_enabled = False
    writeOutput = False

    # Camera mode without a model
    if (len(sys.argv) == 1):
        model_type = None
        video_file = None

    # Camera mode with a model
    elif (len(sys.argv) == 2 and (sys.argv[1] in supported_models)):
        model_type = sys.argv[1]
        video_file = None
    elif (len(sys.argv) == 3 and (sys.argv[1] in supported_models) and (sys.argv[2] == "-l")):
            model_type = sys.argv[1]
            video_file = None
            logs_enabled = True
    elif (len(sys.argv) == 4 and (sys.argv[1] in supported_models) and (sys.argv[2] == "-l") and (sys.argv[3] == "-r")):
            model_type = sys.argv[1]
            video_file = None
            logs_enabled = True
            writeOutput = True
            print("Correct case")

    # Video file mode with a model
    elif (len(sys.argv) == 3 and (sys.argv[1] in supported_models)):
        model_type = sys.argv[1]
        video_file = sys.argv[2]
    elif (len(sys.argv) == 4 and (sys.argv[1] in supported_models) and (sys.argv[3] == "-l")):
            model_type = sys.argv[1]
            video_file = sys.argv[2]
            logs_enabled = True
    elif (len(sys.argv) == 5 and (sys.argv[1] in supported_models) and (sys.argv[3] == "-l") and (sys.argv[4] == "-r")):
            model_type = sys.argv[1]
            video_file = sys.argv[2]
            logs_enabled = True
            writeOutput = True

    else:
        print("\nUsage: <model> <video_filename> \nopt: \n-l :writes logs\n-r :writes video file\nAvailable models are: -ssdm, -ssdmlite, -fasterrcnn, -detr, -yolov5s\nTo just run the webcam provide no args.\n")
        exit()

    if (len(sys.argv) <= 2):
        print("Starting camera ... \nPress q to exit ")
    else:
        print("Starting video ... \nPress q to exit ")
    runProgram(model_type, video_file, logs_enabled, writeOutput)