import cv2 as cv
import numpy as np

def showStats(image, model_type, labels, conf):
    """
    Displays a count of selected objects in the image.
    Recognized objects are:
    * Bicycle   | Label 2
    * Bus       | Label 6
    * Car       | Label 7
    * Motorbike | Label 14
    * Person    | Label 15

    The text is placed below the middle height of the image.

    Args:
    image        -- Opencv image object in RGB
    model_type   -- String with model type
    labels       -- Torch tensor with label indices of varying length.
    conf         -- Torch tensor with confidences for each class.

    Returns:
    Image        -- Opencv image object in RGB with stats
    """
    textColour = (255, 255, 255)
    font = cv.FONT_HERSHEY_SIMPLEX


    labels = labels.numpy()

    # COCO dataset
    if (model_type == "-detr"):
        numCars = np.count_nonzero(labels == 3)
        numTrucksBuses = np.count_nonzero(labels == 6) + np.count_nonzero(labels == 8)
        numMotorCycles = np.count_nonzero(labels == 4)
        numBikes = np.count_nonzero(labels == 2)
        numPed = np.count_nonzero(labels == 1)
        numSign = np.count_nonzero(labels == 10) + np.count_nonzero(labels == 13)
        numOther = 0 # TODO: Other classes
    # Pascal dataset
    else: 
        numCars = np.count_nonzero(labels == 7)
        numTrucksBuses = np.count_nonzero(labels == 6)
        numMotorCycles = np.count_nonzero(labels == 14)
        numBikes = np.count_nonzero(labels == 2)
        numPed = np.count_nonzero(labels == 15)
        numSign = 0 # Not available in PASCAL!
        numOther = 0 # TODO: Other classes


    if (len(conf.numpy()) > 0):
        minConf = '{:.2f}'.format(min(conf.numpy()))
    else:
        minConf = "NA"

    text = "cars: " + str(numCars)
    cv.putText(image, text, (5, int(image.shape[0]/2)+20), font, 0.7, textColour, 1)

    text = "trucks, buses: " + str(numTrucksBuses)
    cv.putText(image, text, (5, int(image.shape[0]/2)+40), font, 0.7, textColour, 1)

    text = "motorcycles: " + str(numMotorCycles)
    cv.putText(image, text, (5, int(image.shape[0]/2)+60), font, 0.7, textColour, 1)

    text = "bicycles: " + str(numBikes)
    cv.putText(image, text, (5, int(image.shape[0]/2)+80), font, 0.7, textColour, 1)

    text = "pedestrians: " + str(numPed)
    cv.putText(image, text, (5, int(image.shape[0]/2)+100), font, 0.7, textColour, 1)

    text = "signs: " + str(numSign)
    cv.putText(image, text, (5, int(image.shape[0]/2)+120), font, 0.7, textColour, 1)

    text = "other: " + str(numOther)
    cv.putText(image, text, (5, int(image.shape[0]/2)+140), font, 0.7, textColour, 1)

    text = "Min confidence: " + minConf
    cv.putText(image, text, (5, int(image.shape[0]/2)+160), font, 0.7, textColour, 1)

    return image