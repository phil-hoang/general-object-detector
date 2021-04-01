"""
Main code to use different models with a webcam or a video file.

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
import time
import sys
from pathlib import Path
import argparse

import cv2 as cv
import torch

from utils2.models import Detection_Model
from visualizer.stats_core import show_stats as show_core_stats
from visualizer.stats_model import show_stats as show_model_stats
import visualizer.signs as signs
import utils2.logger as logger
import utils2.lane_detection as lanes
import utils2.distance as distance


# Required for the slider
def nothing(x):
    pass

#%%
def run_program(model_type, video_file, lane_detection, write_output, enable_logs, sample_number):
    
    # Model selection if chosen in command line
    if model_type != None:
        model = Detection_Model(model_type)
        model.load_model()

    # Prepare input and output
    if (video_file is None):
        # Camera mode
        cap = cv.VideoCapture(0)
        fps = cap.get(cv.CAP_PROP_FPS)
        dim = (int(cap.get(cv.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)) )

        if not cap.isOpened():
            print("ERROR! Cannot open camera")
            exit()
        output_name = "camera-" + str(model_type)

    else:
        # Video mode
        cap = cv.VideoCapture("media/DrivingClips/" + video_file + ".mp4")
        fps = cap.get(cv.CAP_PROP_FPS)
        dim = (int(cap.get(cv.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)))

        if not cap.isOpened():
            print("ERROR! Cannot read video. Does the file exist?")
            exit()

        output_name = video_file

        if model_type != None:
            output_name = output_name + "-" + str(model_type)

        if lane_detection is True:
            output_name = output_name + "-lanes" 

    if write_output is True:
        # Create folder if it doesn't exist
        Path("media/ModelOutputs").mkdir(parents=True, exist_ok=True)
        out = cv.VideoWriter("./media/ModelOutputs/" + output_name 
        + ".avi", cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, dim)

    # Create slider to turn stats and model on or off
    slider = ["Show stats", "Model OFF / ON", "Lanes OFF / ON"]
    if (video_file is None):
        window_name = "Live Detection"

    else:
        window_name = "Video Detection file " + video_file + ".mp4"
    cv.namedWindow(window_name)
    cv.createTrackbar(slider[0], window_name, 1, 1, nothing)    

    if (model_type != None):
        cv.createTrackbar(slider[1], window_name, 1, 1, nothing)
    
    if (lane_detection is True):
        cv.createTrackbar(slider[2], window_name, 1, 1, nothing)
    
    # Load sign symbols
    stop_sign = signs.load()[0]

    # Initialize list for model unrelated core stats. [fps, time.start, time.end]
    stats_core = [None, None, None]

    # Initialize logs
    if enable_logs is True:
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

        # Set slider to turn on or off stats and enable or disable a model
        stats_flag = cv.getTrackbarPos(slider[0], window_name)

        if (model_type != None):
            model_enabled = cv.getTrackbarPos(slider[1], window_name)

        if ((counter % sample_number) == 0):
            counter = 0
            # Locate objects with model if selected
            if ((model_type != None) and (model_enabled == 1)):
                frame, boxes, labels, conf = model.model_predict(image)

            # Lane detection
            if (lane_detection is True):
                lane_enabled = cv.getTrackbarPos(slider[2], window_name)
                if (lane_enabled == 1):
                    frame = lanes.detect(frame)

            # Get time after detection
            stats_core[2] = time.time()
            #  Display stats if selected with slider
            if (stats_flag == 1):
                frame = show_core_stats(frame, stats_core) 

            if ((stats_flag == 1) and (model_type != None) and (model_enabled == 1)):
                frame, model_stats = show_model_stats(frame, model_type, labels, conf)

            # Enable symbols
            if ((model_type != None) and (model_enabled == 1)):
                frame = signs.show_stop_sign(frame, model_type, stop_sign, labels, conf)
            
            # Write logs if enables
            if ((model_type != None) and (enable_logs is True)):
                logs = logger.write_log(logs, stats_core[1], stats_core[2], labels, conf, model_stats)


            # ===================== Estimate distances
            frame = distance.estimate(frame, boxes)    
            # ===================== 



            # Display the resulting frame
            cv.imshow(window_name, frame)
            if ((write_output == True)):
                out.write(frame)

        counter = counter + 1
        if cv.waitKey(1) == ord("q"):
            break
    
    # Writing logs to file
    if enable_logs is True:
        logger.save_logs(logs, video_file, model_type, lane_detection)

    # When everything is done, release the capture
    cap.release()
    if ((write_output is True) and (video_file != None)):
        out.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    # Allow no model or selected model
    supported_models = ["ssdm", "ssdmlite", "detr", "fasterrcnn", "yolov5s"]

    # Parse arguments
    parser = argparse.ArgumentParser(description='Select a model and test with camera or video file')
    parser.add_argument('--model', default=None, choices=supported_models, 
                        help='Select a model')
    parser.add_argument('--f', default=None, help='Path to .mp4 video file')
    parser.add_argument('--sample', default=1, 
                        help="Sets on how many frames detection should be performed. 1 is on all frame, 2 every other etc.")
    parser.add_argument('-lanes', default=False, action='store_const', const=True, 
                        help='Enable lane detection')
    parser.add_argument('-rec', default=False, action='store_const', const=True, 
                        help='Write result to .avi file and log data')
    
    args = parser.parse_args()

    model_type = args.model
    video_file = args.f
    lane_detection = args.lanes
    write_output = args.rec
    enable_logs = args.rec
    sample_number = int(args.sample)
    print(sample_number)

    if (model_type is None and lane_detection is None and enable_logs is True):
        print("Recording and logging without a model is not supported!")
        exit()
  
    if (video_file is False):
        print("Starting camera ... \nPress q to exit ")

    else:
        print("Starting video ... \nPress q to exit ")

    run_program(model_type, video_file, lane_detection, write_output, enable_logs, sample_number)