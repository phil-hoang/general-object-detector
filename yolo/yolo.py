"""
YOLOv3 Model
"""

import torch
import torchvision.transforms as T

def yoloModel():
    """
    Loads the YOLOv3 model from ultralytics
    """

    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).autoshape()
    model.eval()

    return model

def yolo_predict(model, frame, thresh = 0.8):
    """
    Predict with faster rcnn

    Args:
    frame - OpenCV image in BGR

    Return:
    boxes       -- Torch tensor of coordinates of the top left and bottom right of the bounding box ordered as [(x1, y1, x2, y2)]
    labels      -- Torch tensor of index labels for each bounding box [<label indices>]
    scores      -- Torch tensor of class confidence scores for each bounding box [<class scores>]. For COCO, expects 91 different classes 
    """
    

    # Preprocess image
    transform = T.Compose([
    T.ToPILImage(),
    T.Resize((640,640)),
    T.ToTensor(),
    ])

    t_image = transform(frame).unsqueeze(0)
        
    # Predict
    # output is a tuple where the first one has shape [(1, number of bounding box confs, 85 which represents 4 coordinates + number of classes)] and
    # the second has a bunch of gradients or something
    output = model(t_image)
    print(*output[0].shape)

    # Unpack the output into arrays
    boxes = output[0]["boxes"].detach()
    labels = output[0]["labels"].detach()
    conf = output[0]["scores"].detach()
    
    # Threshold results
    keep = conf > thresh

    boxes = boxes[keep]
    labels = labels[keep]
    conf = conf[keep]

    # Resize bounding boxes to match the original size of the image
    img_w = frame.shape[1]/t_image.size()[3]
    img_h = frame.shape[0]/t_image.size()[2]

    boxes = boxes * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)

    """
    print(boxes)
    print(labels)
    print(conf)
    print(t_image.size())
    print(frame.shape)
    """

    return boxes, labels, conf