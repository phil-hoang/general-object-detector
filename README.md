# General Object Detector
Our program that allows you to select which object detection model to use to detect objects.

We currently support the models:
* [SSD](https://arxiv.org/abs/1512.02325) with [MobileNet](https://arxiv.org/abs/1704.04861)
* [Detr](https://arxiv.org/abs/2005.12872)
* [Faster R-CNN](https://arxiv.org/abs/1506.01497)
* [YOLO](https://arxiv.org/abs/1804.02767)

The SSD model is taken from [here](https://github.com/qfgaohao/pytorch-ssd).
The YOLO model is taken from [here](https://github.com/ultralytics/yolov5).


![Example](media/detr-example.jpg)


## Dependencies and setup
* Python ≥ 3.7.10
* OpenCV2
* PyTorch ≥ 1.7.1
* Torchvision ≥ 0.8.2

To use the video function place `.mp4` videos into the folder `media/DrivingClips`.

## Usage
Type 
```
run.py -h
``` 

to see the help.

To use the detector with for example SSD Mobilenet on file video.mp4, type:

```
run.py --model ssdm --f video
```
To use it with the webcam just ommit the filename:
```
run.py --model ssdm
```
Running the command
```
run.py
```
without any arguments just opens the webcam and displays its output.

To enable lane detection use the argument `-lanes`:

```
run.py --model yolov5s -lanes
```


To record the output to a file and write information about the model like the minimum detection confidence or inference time to a log file, use the optional argument `-rec`. Example:

```
run.py --model ssdm --f video -rec
```