"""
Faster R-CNN Model


"""
import torchvision
from torchvision.transforms import ToTensor
import torch
import cv2 as cv


def fasterRcnnModel():
    """
    Loads the Faster R-CNN model from Pytorch

    """
    
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True, progress=True, num_classes=91, 
                                                             pretrained_backbone=True, trainable_backbone_layers=3,
                                                            min_size=360)
    model.eval()

    return model


def predict(model, frame, a, b):
    """
    Predict with faster rcnn

    Args:
    frame - OpenCV image in BGR

    Return:
    boxes
    labels
    probs

    """

    frame_normed = cv.normalize(frame, frame, 0, 255, cv.NORM_MINMAX) 
    frame_normed = ToTensor()(frame_normed)
        
    print(frame_normed.shape)

    im_list = []
    im_list.append(frame_normed)
        
    # Predict and make bounding boxes
    predictions = model(im_list)

    return predictions