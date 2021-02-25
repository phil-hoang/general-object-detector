"""
Main code which provides the functionalities to test different detection models quickly.



"""
import numpy as np
import cv2 as cv
import time
import sys

from ssd_pytorch.ssd import ssdModel as ssd
from visualizer.pascal import drawBoxes as pascalBoxes
from visualizer.stats_model import showStats as statsModel
from visualizer.stats_core import showStats as statsCore


# Required for the slider
def nothing(x):
    pass

#%%
def runProgram():
    
    # TODO
    # Select ssd model if chosen in command line
    if (len(sys.argv) == 2 and (sys.argv[1] == "-ssd")):
        print("Selected ssd")
        ## Load stuff for ssd
        net, predictor, num_classes, class_names = ssd()
    else:
        model_enabled = 0
    
    # Prepare camera
    cap = cv.VideoCapture(0)

    if not cap.isOpened():
        print("ERROR! Cannot open camera")
        exit()

    ## Create slider
    cv.namedWindow('Live Detection')
    switch = 'Model OFF / ON'
    cv.createTrackbar('Show stats', 'Live Detection', 0, 1, nothing)
    if (len(sys.argv) == 2 and (sys.argv[1] == "-ssd")):
        cv.createTrackbar(switch, 'Live Detection', 0, 1, nothing)

    # Initialize list for core stats. [fps, time.start, time.end]
    stats_core = [None, None, None]

    #%% Loop through each frame
    while True:
        # Get time before detection
        stats_core[1] = time.time()
        
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        ## Insert functions here
        # SSD pascal
        stats_core[0] = cap.get(cv.CAP_PROP_FPS)

        statsFlag = cv.getTrackbarPos('Show stats','Live Detection')
        if (len(sys.argv) == 2 and (sys.argv[1] == "-ssd")):
            # If model is selected allow it to be turned on or off
            model_enabled = cv.getTrackbarPos(switch,'Live Detection')
        
        if (len(sys.argv) == 2 and (sys.argv[1] == "-ssd") and model_enabled == 1):
            #if model_enabled == 1:
            boxes, labels, probs = predictor.predict(image, 10, 0.4)
            frame = pascalBoxes(image, probs, boxes, labels, class_names)
        
        # Get time after detection
        stats_core[2] = time.time()
        if (statsFlag == 1):
            frame = statsCore(frame, stats_core) 
        
        if (statsFlag == 1) and (model_enabled == 1):
            frame = statsModel(frame, labels)

        # Display the resulting frame
        cv.imshow('Live Detection', frame)
        if cv.waitKey(1) == ord('q'):
            break

    # When everything is done, release the capture
    cap.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    if ((len(sys.argv) == 2 and (sys.argv[1] != "-ssd")) or (len(sys.argv) > 2)):
        print("Usage: no arg or -<ssd>")
        exit()
    print("Starting camera ... \nPress q to exit ")
    runProgram()