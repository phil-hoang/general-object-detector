"""
Main code which provides the functionalities to test different detection models quickly.

v. 0.01

"""
import numpy as np
import cv2 as cv

from ssd_pytorch import ssdModel as ssd



#%%
def runProgram():
    
    net, predictor = ssd()
    
    
    # Prepare camera
    cap = cv.VideoCapture(0)

    if not cap.isOpened():
        print("ERROR! Cannot open camera")
        exit()

    #%% Loop through each frame
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # Insert functions here





        # Display the resulting frame
        cv.imshow('frame', frame)
        if cv.waitKey(1) == ord('q'):
            break

    # When everything is done, release the capture
    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    print("Starting camera ... \nPress q to exit ")
    runProgram()