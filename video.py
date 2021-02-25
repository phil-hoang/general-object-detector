"""
Code to run different models on an input video. Can write the results to a file if selected.



"""
import numpy as np
import cv2 as cv
import time

from ssd_pytorch.ssd import ssdModel as ssd
from visualizer.pascal import drawBoxes as pascalBoxes
from visualizer.stats_model import showStats as statsModel
from visualizer.stats_core import showStats as statsCore


# Required for the slider
def nothing(x):
    pass

#%%
def runProgram():
    
    ## Load stuff for ssd
    net, predictor, num_classes, class_names = ssd()
    
    # Prepare video input and output
    cap = cv.VideoCapture("dev/driving_scence_yonge_dundas.mp4")
    out = cv.VideoWriter('dev/output.avi', cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), 12.0, (1280, 720))

    if not cap.isOpened():
        print("ERROR! Cannot open camera")
        exit()

    ## Create slider
    cv.namedWindow('Video Detection')
    switch = 'SSD Model'
    cv.createTrackbar('Show stats', 'Video Detection', 0, 1, nothing)
    cv.createTrackbar(switch, 'Video Detection', 0, 1, nothing)

    # Initialize list for core stats. [fps, time.start, time.end]
    stats_core = [None, None, None]

    #%% Loop through each frame
    while True:
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

        ssd_act = cv.getTrackbarPos(switch,'Video Detection')
        statsFlag = cv.getTrackbarPos('Show stats','Video Detection')
        if ssd_act == 1:
            boxes, labels, probs = predictor.predict(image, 10, 0.4)
            frame = pascalBoxes(image, probs, boxes, labels, class_names)
        
        stats_core[2] = time.time()
        if (statsFlag == 1):
            frame = statsCore(frame, stats_core) 
        
        if (statsFlag == 1) and (ssd_act == 1):
            frame = statsModel(frame, labels)


        # Display the resulting frame
        cv.imshow('Video Detection', frame)
        out.write(frame)
        if cv.waitKey(1) == ord('q'):
            break

    # When everything is done, release the capture
    cap.release()
    out.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    print("Starting camera ... \nPress q to exit ")
    runProgram()