import cv2 as cv

def drawBoxes(image, probs, boxes, labels, class_names):
    """
    Draws bounding boxes for selected classes of the Pascal dataset.
    Classes are:
    * Bicycle   | Label 2
    * Bus       | Label 6
    * Car       | Label 7
    * Motorbike | Label 14
    * Person    | Label 15

    Box colours are:
    Motorized vehicles  | Blue
    Bicycles, person    | Red
               

    Args:
    image       - 
    probs       - 
    boxes       - 
    labels      - Torch tensor with label indices
    class_names - Opencv image in BGR

    Returns:
    image       - 
    """
    labelsMotor = [6, 7, 14]
    labelsOther = [2, 15]

    for i in range(boxes.size(0)):
                # Filter for classes
                if (labels[i] in labelsMotor):
                    box = boxes[i, :]
                    label = f"{class_names[labels[i]]}: {probs[i]:.2f}"
                    cv.rectangle(image, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)
                elif (labels[i] in labelsOther):
                    box = boxes[i, :]
                    label = f"{class_names[labels[i]]}: {probs[i]:.2f}"
                    cv.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
                else:
                    pass
    
    # Convert colour channels back to BGR
    image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
    return image