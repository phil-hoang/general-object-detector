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
    labelsKnown = [1,2,3,4,6,8,10,13]

    # COCO dataset
    if (model_type == "-detr" or model_type == "-fasterrcnn"):
        numCars = np.count_nonzero(labels == 3)
        numTrucksBuses = np.count_nonzero(labels == 6) + np.count_nonzero(labels == 8)
        numMotorCycles = np.count_nonzero(labels == 4)
        numBikes = np.count_nonzero(labels == 2)
        numPed = np.count_nonzero(labels == 1)
        numSign = np.count_nonzero(labels == 10) + np.count_nonzero(labels == 13)
        numOther = 0 # TODO: Other classes
        
        # Confidences
        if (len(conf[np.where(labels == 3)])) > 0:
            confCars = '({:.2f})'.format(min(conf[np.where(labels == 3)]))
        else:
            confCars = ""

        if (len(conf[np.where(labels == 6)]) or len(conf[np.where(labels == 8)])) > 0:
            confTrucksBuses = '({:.2f})'.format(min(conf[np.where(labels == 6) or labels == 8]))
        else:
            confTrucksBuses = ""

        if (len(conf[np.where(labels == 4)])) > 0:
            confMotorCycles = '({:.2f})'.format(min(conf[np.where(labels == 4)]))
        else:
            confMotorCycles = ""

        if (len(conf[np.where(labels == 2)])) > 0:
            confBikes = '({:.2f})'.format(min(conf[np.where(labels == 2)]))
        else:
            confBikes = ""

        if (len(conf[np.where(labels == 1)])) > 0:
            confPed = '({:.2f})'.format(min(conf[np.where(labels == 1)]))
        else:
            confPed = ""

        if (len(conf[np.where(labels == 10)]) or len(conf[np.where(labels == 13)])) > 0:
            confSigns = '({:.2f})'.format(min(conf[np.where(labels == 10) or labels == 13]))
        else:
            confSigns = ""

    # Pascal dataset
    else: 
        numCars = np.count_nonzero(labels == 7)
        numTrucksBuses = np.count_nonzero(labels == 6)
        numMotorCycles = np.count_nonzero(labels == 14)
        numBikes = np.count_nonzero(labels == 2)
        numPed = np.count_nonzero(labels == 15)
        numSign = 0 # Not available in PASCAL!
        numOther = 0 # TODO: Other classes


        if (len(conf[np.where(labels == 7)])) > 0:
            confCars = '({:.2f})'.format(min(conf[np.where(labels == 7)]))
        else:
            confCars = ""

        if (len(conf[np.where(labels == 6)])) > 0:
            confTrucksBuses = '({:.2f})'.format(min(conf[np.where(labels == 6)]))
        else:
            confTrucksBuses = ""

        if (len(conf[np.where(labels == 14)])) > 0:
            confMotorCycles = '({:.2f})'.format(min(conf[np.where(labels == 14)]))
        else:
            confMotorCycles = ""

        if (len(conf[np.where(labels == 2)])) > 0:
            confBikes = '({:.2f})'.format(min(conf[np.where(labels == 2)]))
        else:
            confBikes = ""

        if (len(conf[np.where(labels == 15)])) > 0:
            confPed = '({:.2f})'.format(min(conf[np.where(labels == 15)]))
        else:
            confPed = ""

        confSigns = ""


    # Display text
    text = "cars: " + str(numCars) + " " + confCars
    cv.putText(image, text, (5, int(image.shape[0]/2)+20), font, 0.7, textColour, 1)

    text = "trucks, buses: " + str(numTrucksBuses) + " " + confTrucksBuses
    cv.putText(image, text, (5, int(image.shape[0]/2)+40), font, 0.7, textColour, 1)

    text = "motorcycles: " + str(numMotorCycles) + " " + confMotorCycles
    cv.putText(image, text, (5, int(image.shape[0]/2)+60), font, 0.7, textColour, 1)

    text = "bicycles: " + str(numBikes) + " " + confBikes
    cv.putText(image, text, (5, int(image.shape[0]/2)+80), font, 0.7, textColour, 1)

    text = "pedestrians: " + str(numPed) + " " + confPed
    cv.putText(image, text, (5, int(image.shape[0]/2)+100), font, 0.7, textColour, 1)

    text = "signs: " + str(numSign) + " " + confSigns
    cv.putText(image, text, (5, int(image.shape[0]/2)+120), font, 0.7, textColour, 1)

    text = "other: " + str(numOther)
    cv.putText(image, text, (5, int(image.shape[0]/2)+140), font, 0.7, textColour, 1)

    #text = "Min confidence: " + minConf + " " + confCars
    #cv.putText(image, text, (5, int(image.shape[0]/2)+160), font, 0.7, textColour, 1)

    return image