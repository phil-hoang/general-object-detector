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
    model_stats  -- Dictionary with stats. Keys: numCars, confMinCars, confMaxCars, ...
    """
    textColour = (255, 255, 255)
    font = cv.FONT_HERSHEY_SIMPLEX


    labels = labels.numpy()
    labelsKnown = [1,2,3,4,6,8,10,13]
    keys = ["numCars", "confMinCars", "confMaxCars",
            "numTrucksBuses", "confMinTrucksBuses", "confMaxTrucksBuses",
            "numMotorCycles", "confMinMotorCycles", "confMaxMotorCycles",
            "numBikes", "confMinBikes", "confMaxBikes",
            "numPed", "confMinPed", "confMaxPed",
            "numSign", "confMinSign", "confMaxSign"]
    model_stats = dict.fromkeys(keys)

    # COCO dataset
    if (model_type == "-detr" or model_type == "-fasterrcnn" or model_type == "-yolov5s"):
        model_stats["numCars"] = np.count_nonzero(labels == 3)
        model_stats["numTrucksBuses"] = np.count_nonzero(labels == 6) + np.count_nonzero(labels == 8)
        model_stats["numMotorCycles"] = np.count_nonzero(labels == 4)
        model_stats["numBikes"] = np.count_nonzero(labels == 2)
        model_stats["numPed"] = np.count_nonzero(labels == 1)
        model_stats["numSign"] = np.count_nonzero(labels == 10) + np.count_nonzero(labels == 13)
        model_stats["numOther"] = 0 # TODO: Other classes
        
        # Confidences
        if (len(conf[np.where(labels == 3)])) > 0:
            model_stats["confMinCars"] = min(conf[np.where(labels == 3)])
            model_stats["confMaxCars"] = max(conf[np.where(labels == 3)])
            confCars = '({:.2f}, {:.2f})'.format(model_stats["confMinCars"], model_stats["confMaxCars"])
        else:
            model_stats["confMinCars"] = None
            model_stats["confMaxCars"] = None
            confCars = ""

        if (len(conf[np.where(labels == 6)]) or len(conf[np.where(labels == 8)])) > 0:
            res = np.append(np.where(labels == 6), np.where(labels == 8))
            model_stats["confMinTrucksBuses"] = min(conf[res])
            model_stats["confMaxTrucksBuses"] = max(conf[res])
            confTrucksBuses = '({:.2f}, {:.2f})'.format(model_stats["confMinTrucksBuses"], model_stats["confMaxTrucksBuses"])
        else:
            model_stats["confMinTrucksBuses"] = None
            model_stats["confMaxTrucksBuses"] = None
            confTrucksBuses = ""

        if (len(conf[np.where(labels == 4)])) > 0:
            model_stats["confMinMotorCycles"] = min(conf[np.where(labels == 4)])
            model_stats["confMaxMotorCycles"] = max(conf[np.where(labels == 4)])
            confMotorCycles = '({:.2f}, {:.2f})'.format(model_stats["confMinMotorCycles"], model_stats["confMaxMotorCycles"])
        else:
            model_stats["confMinMotorCycles"] = None
            model_stats["confMaxMotorCycles"] = None
            confMotorCycles = ""

        if (len(conf[np.where(labels == 2)])) > 0:
            model_stats["confMinBikes"] = min(conf[np.where(labels == 2)])
            model_stats["confMaxBikes"] = max(conf[np.where(labels == 2)])
            confBikes = '({:.2f}, {:.2f})'.format(model_stats["confMinBikes"], model_stats["confMaxBikes"])
        else:
            model_stats["confMinBikes"] = None
            model_stats["confMaxBikes"] = None
            confBikes = ""

        if (len(conf[np.where(labels == 1)])) > 0:
            model_stats["confMinPed"] = min(conf[np.where(labels == 1)])
            model_stats["confMaxPed"] = max(conf[np.where(labels == 1)])
            confPed = '({:.2f}, {:.2f})'.format(model_stats["confMinPed"], model_stats["confMaxPed"])
        else:
            model_stats["confMinPed"] = None
            model_stats["confMaxPed"] = None
            confPed = ""

        if (len(conf[np.where(labels == 10)]) or len(conf[np.where(labels == 13)])) > 0:
            res = np.append(np.where(labels == 10), np.where(labels == 13))
            model_stats["confMinSigns"] = min(conf[res])    
            model_stats["confMaxSigns"] = max(conf[res])
            confSigns = '({:.2f}, {:.2f})'.format(model_stats["confMinSigns"], model_stats["confMaxSigns"])
        else:
            model_stats["confMinSigns"] = None
            model_stats["confMaxSigns"] = None
            confSigns = ""

    # Pascal dataset
    else: 
        model_stats["numCars"] = np.count_nonzero(labels == 7)
        model_stats["numTrucksBuses"] = np.count_nonzero(labels == 6)
        model_stats["numMotorCycles"] = np.count_nonzero(labels == 14)
        model_stats["numBikes"] = np.count_nonzero(labels == 2)
        model_stats["numPed"] = np.count_nonzero(labels == 15)
        model_stats["numSign"] = 0 # Not available in PASCAL!
        model_stats["numOther"] = 0 # TODO: Other classes


        if (len(conf[np.where(labels == 7)])) > 0:
            model_stats["confMinCars"] = min(conf[np.where(labels == 7)])
            model_stats["confMaxCars"] = max(conf[np.where(labels == 7)])
            confCars = '({:.2f}, {:.2f})'.format(model_stats["confMinCars"], model_stats["confMaxCars"])
        else:
            model_stats["confMinCars"] = None
            model_stats["confMaxCars"] = None
            confCars = ""

        if (len(conf[np.where(labels == 6)])) > 0:
            model_stats["confMinTrucksBuses"] = min(conf[np.where(labels == 6)])
            model_stats["confMaxTrucksBuses"] = max(conf[np.where(labels == 6)])
            confTrucksBuses = '({:.2f}, {:.2f})'.format(model_stats["confMinTrucksBuses"], model_stats["confMaxTrucksBuses"])
        else:
            model_stats["confMinTrucksBuses"] = None
            model_stats["confMaxTrucksBuses"] = None
            confTrucksBuses = ""

        if (len(conf[np.where(labels == 14)])) > 0:
            model_stats["confMinMotorCycles"] = min(conf[np.where(labels == 14)])
            model_stats["confMaxMotorCycles"] = max(conf[np.where(labels == 14)])
            confMotorCycles = '({:.2f}, {:.2f})'.format(model_stats["confMinMotorCycles"], model_stats["confMaxMotorCycles"])
        else:
            model_stats["confMinMotorCycles"] = None
            model_stats["confMaxMotorCycles"] = None
            confMotorCycles = ""

        if (len(conf[np.where(labels == 2)])) > 0:

            model_stats["confMinBikes"] = min(conf[np.where(labels == 2)])
            model_stats["confMaxBikes"] = max(conf[np.where(labels == 2)])
            confBikes = '({:.2f}, {:.2f})'.format(model_stats["confMinBikes"], model_stats["confMaxBikes"])
        else:
            model_stats["confMinBikes"] = None
            model_stats["confMaxBikes"] = None
            confBikes = ""

        if (len(conf[np.where(labels == 15)])) > 0:
            #confPed = '({:.2f})'.format(min(conf[np.where(labels == 1)]))
            model_stats["confMinPed"] = min(conf[np.where(labels == 15)])
            model_stats["confMaxPed"] = max(conf[np.where(labels == 15)])
            confPed = '({:.2f}, {:.2f})'.format(model_stats["confMinPed"], model_stats["confMaxPed"])
        else:
            model_stats["confMinPed"] = None
            model_stats["confMaxPed"] = None
            confPed = ""
        
        # No signs in Pascal
        model_stats["confMinSigns"] = None
        model_stats["confMaxSigns"] = None
        confSigns = ""

    # Display text
    text = "cars: " + str(model_stats["numCars"]) + " " + confCars
    cv.putText(image, text, (5, int(image.shape[0]/2)+20), font, 0.7, textColour, 1)

    text = "trucks, buses: " + str(model_stats["numTrucksBuses"]) + " " + confTrucksBuses
    cv.putText(image, text, (5, int(image.shape[0]/2)+40), font, 0.7, textColour, 1)

    text = "motorcycles: " + str(model_stats["numMotorCycles"]) + " " + confMotorCycles
    cv.putText(image, text, (5, int(image.shape[0]/2)+60), font, 0.7, textColour, 1)

    text = "bicycles: " + str(model_stats["numBikes"]) + " " + confBikes
    cv.putText(image, text, (5, int(image.shape[0]/2)+80), font, 0.7, textColour, 1)

    text = "pedestrians: " + str(model_stats["numPed"]) + " " + confPed
    cv.putText(image, text, (5, int(image.shape[0]/2)+100), font, 0.7, textColour, 1)

    text = "signs: " + str(model_stats["numSign"]) + " " + confSigns
    cv.putText(image, text, (5, int(image.shape[0]/2)+120), font, 0.7, textColour, 1)

    text = "other: " + str(model_stats["numOther"])
    cv.putText(image, text, (5, int(image.shape[0]/2)+140), font, 0.7, textColour, 1)

    #text = "Min confidence: " + minConf + " " + confCars
    #cv.putText(image, text, (5, int(image.shape[0]/2)+160), font, 0.7, textColour, 1)

    return image, model_stats