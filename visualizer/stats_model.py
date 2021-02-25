import cv2 as cv
import numpy as np

def showStats(image, stats):
    """
    Adds stats to the side of the image

    Args:
    image   - Input image
    stats   - List with stats to show

    Returns:
    image   - Image with stats
    """

    stats = stats.numpy()
    numCars = np.count_nonzero(stats == 7)
    numTrucksBuses = np.count_nonzero(stats == 6)
    numMotorCycles = np.count_nonzero(stats == 14)
    numBikes = np.count_nonzero(stats == 2)
    numPed = np.count_nonzero(stats == 15)
    numSign = 0 # Not available in PASCAL!

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


    return image