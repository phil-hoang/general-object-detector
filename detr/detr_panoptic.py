import torchvision.transforms as T
import torch
import numpy
import io
import cv2 as cv

from PIL import Image
from copy import deepcopy
from detectron2.utils.visualizer import Visualizer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog

import panopticapi
from panopticapi.utils import id2rgb, rgb2id

def detr_panoptic_load():
    """
    Loads the detr model using resnet50 for panoptic segmentation

    Returns: the detr model pretrained on COCO dataset
    """
    model, postprocessor = torch.hub.load('facebookresearch/detr', 'detr_resnet50_panoptic', pretrained=True, return_postprocessor=True, num_classes=250)
    model.eval()

    return model, postprocessor


def detr_panoptic_predict(model, postprocessor, image):
    """
    Function used to preprocess the image, feed it into the panoptic detr model, and prepare the output draw bounding boxes.
    Outputs are thresholded.
    Related functions: detr_load, draw_boxes in coco.py

    Args: 
    model       -- the detr model from detr_load()
    image       -- Array the original image from openCV [width, height, channels]

    Returns: 
    boxes       -- Torch tensor of coordinates of the top left and bottom right of the bounding box ordered as [(x1, y1, x2, y2)]
    labels      -- Torch tensor of index labels for each bounding box [<label indices>]
    scores      -- Torch tensor of class confidence scores for each bounding box [<class scores>]. For COCO, expects 91 different classes 
    """

    # Preprocess image
    transform = T.Compose([
    T.ToPILImage(),
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    pil_image = T.ToPILImage()(image)

    t_image = transform(image).unsqueeze(0)

    # Run image through detr model
    output = model(t_image)

    # compute the scores, excluding the "no-object" class (the last one)
    scores = output["pred_logits"].softmax(-1)[..., :-1].max(-1)[0]

    # the post-processor expects as input the target size of the predictions (which we set here to the image size)
    result = postprocessor(output, torch.as_tensor(t_image.shape[-2:]).unsqueeze(0))[0]

    # We extract the segments info and the panoptic result from DETR's prediction
    segments_info = deepcopy(result["segments_info"])
    # Panoptic predictions are stored in a special format png
    panoptic_seg = Image.open(io.BytesIO(result['png_string']))
    final_w, final_h = panoptic_seg.size
    # We convert the png into an segment id map
    panoptic_seg = numpy.array(panoptic_seg, dtype=numpy.uint8)
    panoptic_seg = torch.from_numpy(rgb2id(panoptic_seg))

        
    # Detectron2 uses a different numbering of coco classes, here we convert the class ids accordingly
    meta = MetadataCatalog.get("coco_2017_val_panoptic_separated")
    for i in range(len(segments_info)):
        c = segments_info[i]["category_id"]
        segments_info[i]["category_id"] = meta.thing_dataset_id_to_contiguous_id[c] if segments_info[i]["isthing"] else meta.stuff_dataset_id_to_contiguous_id[c]


    # Finally we visualize the prediction

    v = Visualizer(numpy.array(pil_image.resize((final_w, final_h)))[:, :, ::-1], meta, scale=1.0)
    v._default_font_size = 20
    v = v.draw_panoptic_seg_predictions(panoptic_seg, segments_info, area_threshold=0)
    # cv.imshow("Live Detection", v.get_image())

    return v.get_image()