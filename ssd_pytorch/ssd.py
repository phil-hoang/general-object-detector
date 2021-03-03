"""
Loads the SSD model from https://github.com/qfgaohao/pytorch-ssd.


"""

from ssd_pytorch.vision.ssd.mobilenetv1_ssd import  create_mobilenetv1_ssd, create_mobilenetv1_ssd_predictor
from ssd_pytorch.vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite, create_mobilenetv2_ssd_lite_predictor
from ssd_pytorch.vision.ssd.vgg_ssd import create_vgg_ssd, create_vgg_ssd_predictor

def ssdModel(extractor):
    """
    Builds the ssd model
    Args:
    extractor   - String to specify which backbone to use

    Returns:
    net         - 
    predictor   -
    """

    label_path = "ssd_pytorch/models/voc-model-labels.txt"
    class_names = [name.strip() for name in open(label_path).readlines()]
    num_classes = len(class_names)

    if (extractor == "-ssdm"):
        print("Using feature extractor: mobilnet")
        model_path = "ssd_pytorch/models/mobilenet-v1-ssd-mp-0_675.pth"
        
        # Load SSD net with mobilenet feature extractor
        net = create_mobilenetv1_ssd(len(class_names), is_test=True)
        net.load(model_path)
        predictor = create_mobilenetv1_ssd_predictor(net, candidate_size=200)

    if (extractor == "-ssdmlite"):
        print("Using feature extractor: mobilnet lite")
        model_path = "ssd_pytorch/models/mb2-ssd-lite-mp-0_686.pth"
        
        # Load SSD net with mobilenet lite feature extractor
        net = create_mobilenetv2_ssd_lite(len(class_names), is_test=True)
        net.load(model_path)
        predictor = create_mobilenetv2_ssd_lite_predictor(net, candidate_size=200)

    if (extractor == "-ssdvgg"):
        print("Using feature extractor: vgg")
        model_path = "ssd_pytorch/models/vgg16_reducedfc.pth"
        
        # Load SSD net with mobilenet lite feature extractor
        net = create_vgg_ssd(len(class_names), is_test=True)
        net.load(model_path)
        predictor = create_vgg_ssd_predictor(net, candidate_size=200)

    return net, predictor