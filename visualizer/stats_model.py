import cv2 as cv
import numpy as np

def showStats(image, stats, conf):
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
    image - Opencv image object in RGB
    stats - Torch tensor with label indices of varying length.
    conf  - Torch tensor with confidences for each class.

    Returns:
    Image - Opencv image object in RGB with stats
    """

    stats = stats.numpy()
    numCars = np.count_nonzero(stats == 7)
    numTrucksBuses = np.count_nonzero(stats == 6)
    numMotorCycles = np.count_nonzero(stats == 14)
    numBikes = np.count_nonzero(stats == 2)
    numPed = np.count_nonzero(stats == 15)
    numSign = 0 # Not available in PASCAL!
    numOther = 0 # Other classes


    if (len(conf.numpy()) > 0):
        minConf = '{:.2f}'.format(min(conf.numpy()))
    else:
        minConf = "NA"

    text = "cars: " + str(numCars)
    cv.putText(image, text, (5, int(image.shape[0]/2)+20), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

    text = "trucks, buses: " + str(numTrucksBuses)
    cv.putText(image, text, (5, int(image.shape[0]/2)+40), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

    text = "motorcycles: " + str(numMotorCycles)
    cv.putText(image, text, (5, int(image.shape[0]/2)+60), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

    text = "bicycles: " + str(numBikes)
    cv.putText(image, text, (5, int(image.shape[0]/2)+80), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

    text = "pedestrians: " + str(numPed)
    cv.putText(image, text, (5, int(image.shape[0]/2)+100), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

    text = "signs: " + str(numSign)
    cv.putText(image, text, (5, int(image.shape[0]/2)+120), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

    text = "other: " + str(numOther)
    cv.putText(image, text, (5, int(image.shape[0]/2)+140), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

    text = "Min confidence: " + minConf
    cv.putText(image, text, (5, int(image.shape[0]/2)+150), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

    return image