import cv2 as cv

def drawBoxes(image, probs, boxes, labels, class_names):
    """
    Draws bounding boxes for the Pascal dataset.

    Args:
    image       - 
    probs       - 
    boxes       - 
    labels      - Torch tensor with label indices
    class_names - Opencv image in BGR

    Returns:
    image       - 
    """

    for i in range(boxes.size(0)):
                box = boxes[i, :]
                label = f"{class_names[labels[i]]}: {probs[i]:.2f}"
                cv.rectangle(image, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)
    
    # Convert colour channels back to BGR
    image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
    return image