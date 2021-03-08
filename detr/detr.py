import torchvision.transforms as T
import torch

"""
Functions for the detr object detection model

"""

def detr_load():
    """
    Loads the detr model using resnet50

    Returns: the detr model pretrained on COCO dataset
    """

    model = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=True)
    model.eval()

    return model

def detr_predict(model, image):
    """
    Function used preprocess the image, feed it into the detr model, and prepare the output draw bounding boxes

    Inputs: model - the detr model from detr_load()
            image - Array the original image from openCV [width, height, channels]

    Outputs: predictions - Dictionary with boxes, labels and scores. Not a list of dict! Sorted for each bounding box
                boxes   - List of coordinates of the top left and bottom right of the bounding box ordered as [(x1, y1, x2, y2)]
                labels  - List of index labels for each bounding box [<label indices>]
                scores  - List of class confidence scores for each bounding box [<class scores>]. For COCO, expects 91 different classes.
    
    Related functions: detr_load, draw_boxes in coco.py
    """
    def box_cxcywh_to_xyxy(x):
        # Converts bounding boxes to (x1, y1, x2, y2) coordinates of top left and bottom right corners
        x_c, y_c, w, h = x.unbind(1)
        b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
            (x_c + 0.5 * w), (y_c + 0.5 * h)]
        return torch.stack(b, dim=1)

    def rescale_bboxes(out_bbox, size):
        # Scale the bounding boxes to the image size
        img_w, img_h = size
        b = box_cxcywh_to_xyxy(out_bbox)
        b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
        return b

    # Preprocess image
    transform = T.Compose([
    T.ToPILImage(),
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    t_image = transform(image).unsqueeze(0)

    # output is a dict containing "pred_logits" of [batch_size x num_queries x (num_classes + 1)]
    # and "pred_boxes" of shape (center_x, center_y, height, width) normalized to be between [0, 1]
    output = model(t_image)

    # Scale the class probabilities to add up to 1
    probas = output['pred_logits'].softmax(-1)[0,:,:-1]

    # Create a dict for outputs
    boxes = rescale_bboxes(output['pred_boxes'][0], (t_image.size()[3], t_image.size()[2])).detach()

    predictions = { 'boxes' : boxes,
                    'scores' : probas.max(-1).values,
                    'labels' : probas.max(-1).indices}

    return predictions

