import argparse
import sys


if __name__ == '__main__':
    # Allow no model or selected model
    # TODO: Change subsequent usage of the model arg, since - is removed!
    supported_models = ["-ssdm", "-ssdmlite", "-detr", "-fasterrcnn", "-yolov5s"]
    
    # Initialize log and write output variables
    logs_enabled = False
    lane_detection = False
    writeOutput = False
    video_file = None
    model_type = None

    parser = argparse.ArgumentParser(description='Select a model and test with camera or video file')
    parser.add_argument('--m', default=False,help='Select a model. Available: <yolov5s>, <...>')
    parser.add_argument('--f', default=False,help='Path to .mp4 video file')
    parser.add_argument('-l', default=False, action='store_const', const=True, help='Enable lane detection')
    parser.add_argument('-r', default=False, action='store_const', const=True, help='Write result to .avi file and log data')
    args = parser.parse_args()

    videoFile = args.f
    modelType = args.m
    laneDetection = args.l
    writeOutput = args.r
    enableLogs = args.r
    
    


    print("------")
    print("Model: " + str(modelType))
    print("File: " + str(videoFile))
    print("Lanes: " + str(laneDetection))
    print("Output: " + str(writeOutput))
    print("Logs: " + str(enableLogs))