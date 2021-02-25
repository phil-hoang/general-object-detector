"""
SSD model from ...

"""

from ssd_pytorch.vision.ssd.mobilenetv1_ssd import  create_mobilenetv1_ssd, create_mobilenetv1_ssd_predictor


def ssdModel():
    """
    Builds the ssd model

    """

    model_path = "ssd_pytorch/models/mobilenet-v1-ssd-mp-0_675.pth"
    label_path = "ssd_pytorch/models/voc-model-labels.txt"
    class_names = [name.strip() for name in open(label_path).readlines()]
    num_classes = len(class_names)

    # Load SSD net
    net = create_mobilenetv1_ssd(len(class_names), is_test=True)
    net.load(model_path)
    predictor = create_mobilenetv1_ssd_predictor(net, candidate_size=200)

    return net, predictor, num_classes, class_names