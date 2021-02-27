"""
Code to run different models on an input video. Can write the results to a file if selected.



"""
import numpy as np
import cv2 as cv
import time
import sys

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
def runProgram():
    
    #%% Model selection if chosen in command line
    if ( (len(sys.argv) == 3) and (model_type == "-ssdm")):
        net, predictor = ssd("-ssdm")
    elif ( (len(sys.argv) == 3) and (model_type == "-ssdmlite")):
        net, predictor = ssd("-ssdmlite")
    elif ( (len(sys.argv) == 3) and (model_type == "-fasterrcnn")):
            predictor = frcnn()
    else:
        model_enabled = 0
    
    # Prepare video input and output
    selected_video = sys.argv[2]
    cap = cv.VideoCapture("dev/DrivingClips/" + selected_video + ".mp4")
    #out = cv.VideoWriter('dev/output.avi', cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), 12.0, (1280, 720))

    if not cap.isOpened():
        print("ERROR! Cannot open camera")
        exit()

    # Create slider to turn stats and model on or off
    cv.namedWindow('Video Detection')
    switch = 'Model OFF / ON'
    cv.createTrackbar('Show stats', 'Video Detection', 1, 1, nothing)
    if (len(sys.argv) == 3):
        cv.createTrackbar(switch, 'Video Detection', 1, 1, nothing)

    # Load sign symbols
    stop_sign = signs.load()[0]

    # Initialize list for model unrelated core stats. [fps, time.start, time.end]
    stats_core = [None, None, None]

    #%% Loop through each frame
    while True:
        stats_core[1] = time.time()
        
        # Get a frame, convert to RGB and get frames per second fps
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        stats_core[0] = cap.get(cv.CAP_PROP_FPS)

        # Set slider to turn on or off stats and enable or disable a model, if a model is selected
        statsFlag = cv.getTrackbarPos('Show stats','Video Detection')
        if (len(sys.argv) == 3):
            model_enabled = cv.getTrackbarPos(switch,'Video Detection')

        # Locate objects with model if selected
        if (len(sys.argv) == 3 and model_enabled == 1):
            boxes, labels, probs = predictor.predict(image, 10, 0.4)
            frame = pascalBoxes(image, probs, boxes, labels)
            
        # Get time after detection
        stats_core[2] = time.time()

        #  Display stats if selected with slider
        if (statsFlag == 1):
            frame = showCoreStats(frame, stats_core) 
        if (statsFlag == 1) and (model_enabled == 1):
            frame = showModelStats(frame, labels)

        # Enable symbols
        if (model_enabled == 1):
            frame = signs.showStopSign(frame, stop_sign, labels, probs)
        
        # Display the resulting frame
        cv.imshow('Video Detection', frame)
        #out.write(frame)
        if cv.waitKey(1) == ord('q'):
            break

    # When everything is done, release the capture
    cap.release()
    #out.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    # Allow no model or selected model
    if (len(sys.argv) == 2 ):
        model_type = None
    elif (len(sys.argv) == 3 and (sys.argv[1] == "-ssdm")):
        model_type = "-ssdm"
    elif (len(sys.argv) == 3 and (sys.argv[1] == "-ssdmlite")):
        model_type = "-ssdmlite"
    elif (len(sys.argv) == 3 and (sys.argv[1] == "-fasterrcnn")):
        model_type = "-fasterrcnn"
    else:
        print("Usage: no arg or -ssdm or -ssdmlite or -fasterrcnn and video_name")
        exit()
    
    print("Starting video ... \nPress q to exit ")
    runProgram()