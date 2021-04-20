from ssd_pytorch.ssd import ssd_model as ssd
from detr.detr import detr_load as detr
from detr.detr import detr_predict
from detr.detr_panoptic import detr_panoptic_load as detr_panoptic
from detr.detr_panoptic import detr_panoptic_predict
from faster_rcnn.fasterrcnn import faster_rcnn_model as frcnn
from faster_rcnn.fasterrcnn import frcnn_predict
from yolo.yolo import yolo_model as yolo
from yolo.yolo import yolo_predict
from visualizer.pascal import draw_boxes as pascal_boxes
from visualizer.coco import draw_boxes as coco_boxes

class Detection_Model:
    def __init__(self, model_type):
        if model_type:
            self.model_type = model_type
            self.predictor = None
            

    def load_model(self, model_type = None):
        if model_type:
            self.model_type = model_type

        if ((self.model_type == "ssdm") or (self.model_type == "ssdmlite")):
            _, self.predictor = ssd(self.model_type)
        elif (self.model_type == "fasterrcnn"):
            self.predictor = frcnn()
        elif (self.model_type == "detr"):
            self.predictor = detr()
        elif (self.model_type == "yolov5s"):
            self.predictor = yolo()
        elif (self.model_type == "detrpanoptic"):
            self.predictor, self.postprocessor = detr_panoptic()
        else:
            print("Unable to load model. Valid models include ssdm, ssdmlite, fasterrcnn, detr, and yolov5s")


    def model_predict(self, image):
        
        if (self.model_type == "ssdm" or self.model_type == "ssdmlite"):
            boxes, labels, conf = self.predictor.predict(image, 10, 0.4)
            frame = pascal_boxes(image, conf, boxes, labels)
        elif (self.model_type == "detr"):
            boxes, labels, conf = detr_predict(self.predictor, image)
            frame = coco_boxes(image, boxes, labels, conf)
        elif (self.model_type == "fasterrcnn"):
            boxes, labels, conf = frcnn_predict(self.predictor, image)
            frame = coco_boxes(image, boxes, labels, conf)
        elif (self.model_type == "yolov5s"):
            boxes, labels, conf = yolo_predict(self.predictor, image)
            frame = coco_boxes(image, boxes, labels, conf)
        elif (self.model_type == "detrpanoptic"):
            frame = detr_panoptic_predict(self.predictor, self.postprocessor, image)
            return frame, None, None, None
        else:
            raise Exception("Error: Model not loaded")
        
        return frame, boxes, labels, conf