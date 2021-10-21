"""
YOLOv3 Model
"""
import numpy as np
import torch
import torch.nn as nn
from torch import jit
import torchvision.transforms as T
from torchvision.ops import nms

def coco80_to_coco91_class(label):  
    # converts 80-index (val2014) to 91-index (paper)  
    coco91_classes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
                35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
                64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90, 91]

    x = [coco91_classes[i] for i in label]
    x = torch.tensor(x, dtype=torch.long)

    return x


def yolo_to_coco_traffic(label):
    """
    Converts 0-index yolo data to custom COCO traffic data.
    The custom dataset has the same labels as COCO, with the extensions 92,93 and 94.
    """

    traffic_classes = np.arange(1, 15)
    x = [traffic_classes[i] for i in label]

    # Map traffic labels to COCO label.
    traffic_to_coco = {1:1 , 2:2 ,3:3 ,4:4 ,5:6 ,6:7 ,7:8 ,8:11 , 9:13 , 10:17 ,11:18 , 12:92 , 13:93 , 14:94}
    x = [traffic_to_coco[i] for i in x]
    x = torch.tensor(x, dtype=torch.long)

    return x


def yolo_model():
    """
    Loads the YOLOv5 model from ultralytics
    """

    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, force_reload = True)
    model.eval()

    return model

def yolo_model_traffic():
    """
    Loads the custom YOLOv5 model. It has to be placed into /yolo.
    """

    weights = 'yolo/yolov5sTraffic.pt'
    model = torch.hub.load('ultralytics/yolov5', 'custom', weights, force_reload=True)
    model.eval()

    return model

def yolo_predict(model, frame, thresh = 0.6):
    """
    Predict with yolo model

    Args:
    frame - OpenCV image in BGR

    Return:
    boxes       -- Torch tensor of coordinates of the top left and bottom right of the bounding box ordered as [(x1, y1, x2, y2)]
    labels      -- Torch tensor of index labels for each bounding box [<label indices>]
    scores      -- Torch tensor of class confidence scores for each bounding box [<class scores>]. For COCO, expects 91 different classes 
    """

    # Predict
    output = model(frame)

    # Unpack the output
    result = output.xyxy[0]
    
    boxes = result[:,:4]
    conf = result[:,4]
    labels = result[:,5].type(torch.LongTensor)

    # Apply threshold
    keep = conf > thresh
    boxes = boxes[keep]
    conf = conf[keep]
    labels = labels[keep]

    # Convert COCO labels because some classes were removed
    labels = coco80_to_coco91_class(labels)

    return boxes, labels, conf


def yolo_traffic_predict(model, frame, thresh = 0.6):
    """
    Predict with yolo model trained to detect traffic light status and more.

    Args:
    frame - OpenCV image in BGR

    Return:
    boxes       -- Torch tensor of coordinates of the top left and bottom right of the bounding box ordered as [(x1, y1, x2, y2)]
    labels      -- Torch tensor of index labels for each bounding box [<label indices>]
    scores      -- Torch tensor of class confidence scores for each bounding box [<class scores>]. For COCO, expects 91 different classes 
    """

    # Predict
    output = model(frame)

    # Unpack the output
    result = output.xyxy[0]
    
    boxes = result[:,:4]
    conf = result[:,4]
    labels = result[:,5].type(torch.LongTensor)

    # Apply threshold
    keep = conf > thresh
    boxes = boxes[keep]
    conf = conf[keep]
    labels = labels[keep] # In 0-indexed yolo format

    # Convert COCO labels because some classes were removed
    labels = yolo_to_coco_traffic(labels)

    return boxes, labels, conf
