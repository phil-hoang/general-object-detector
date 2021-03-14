import cv2 as cv
import numpy as np
from utils2 import constants
from visualizer import coco, pascal


def getMinMaxConf(conf, labels, label, model_stats, statsMin, statsMax):
    """
    Gets the min and max confidence for the given class.

    Args:
    conf        -- Torch tensor with confidences for each class.
    labels      -- Torch tensor with label indices of varying length.
    label       -- Integer with class label, as a dict key.
    model_stats -- Dictionary with stats. Keys: numCars, confMinCars, confMaxCars, ... 
    statsMin    -- String for min value name.
    statsMax    -- String for max value name.

    Returns:
    model_stats -- Dict with the updated stats.
    confClass   -- String with the formatted min and max confidence.
    """
    if (len(conf[np.where(labels == label)])) > 0: 
        model_stats[statsMin] = min(conf[np.where(labels == label)])
        model_stats[statsMax] = max(conf[np.where(labels == label)])
        confClass = '({:.2f}, {:.2f})'.format(model_stats[statsMin], model_stats[statsMax])
    else:
        model_stats[statsMin] = None
        model_stats[statsMax] = None
        confClass = ""

    return model_stats, confClass


def showStats(image, model_type, labels, conf):
    """
    Displays a count of selected objects in the image for both COCO and PASCAL data.
    
    Args:
    image        -- Opencv image object in RGB
    model_type   -- String with model type
    labels       -- Torch tensor with label indices of varying length.
    conf         -- Torch tensor with confidences for each class.

    Returns:
    Image        -- Opencv image object in RGB with stats
    model_stats  -- Dictionary with stats. Keys: numCars, confMinCars, confMaxCars, ...
    """

    # Extract labels
    labels = labels.numpy()

    # Initialize model_stats dictionary
    model_stats = constants.modelStats()

    # COCO dataset
    cocoModels = coco.supportedModels()
        
    if (model_type in cocoModels):
        # Get labels
        cocoLabels = coco.labels()

        labelList = ["car", "motorcycle", "bicycle", "person"]
        statList = ["numCars", "numMotorCycles", "numBikes", "numPed"]

        for i in range(4):
            model_stats[statList[i]] = np.count_nonzero(labels == cocoLabels[labelList[i]])

        model_stats["numTrucksBuses"] = np.count_nonzero(labels == cocoLabels["truck"]) + np.count_nonzero(labels == cocoLabels["bus"])
        model_stats["numSign"] = np.count_nonzero(labels == cocoLabels["stopsign"]) + np.count_nonzero(labels == cocoLabels["stoplight"])

        # Extract confidences abd 
        model_stats, confCars = getMinMaxConf(conf, labels, cocoLabels["car"], model_stats, "confMinCars", "confMaxCars")
        model_stats, confMotorCycles = getMinMaxConf(conf, labels, cocoLabels["motorcycle"], model_stats, "confMinMotorCycles", "confMaxMotorCycles")
        model_stats, confBikes = getMinMaxConf(conf, labels, cocoLabels["bicycle"], model_stats, "confMinBikes", "confMaxBikes")
        model_stats, confPed = getMinMaxConf(conf, labels, cocoLabels["person"], model_stats, "confMinPed", "confMaxPed")

        # Confidence for summarized classes. Currently they have to be handled seperatly.
        if (len(conf[np.where(labels == cocoLabels["truck"])]) or len(conf[np.where(labels == cocoLabels["bus"])])) > 0:
            res = np.append(np.where(labels == cocoLabels["truck"]), np.where(labels == cocoLabels["bus"]))
            model_stats["confMinTrucksBuses"] = min(conf[res])
            model_stats["confMaxTrucksBuses"] = max(conf[res])
            confTrucksBuses = '({:.2f}, {:.2f})'.format(model_stats["confMinTrucksBuses"], model_stats["confMaxTrucksBuses"])
        else:
            model_stats["confMinTrucksBuses"] = None
            model_stats["confMaxTrucksBuses"] = None
            confTrucksBuses = ""

        if (len(conf[np.where(labels == cocoLabels["stopsign"])]) or len(conf[np.where(labels == cocoLabels["stoplight"])])) > 0:
            res = np.append(np.where(labels == cocoLabels["stopsign"]), np.where(labels == cocoLabels["stoplight"]))
            model_stats["confMinSigns"] = min(conf[res])    
            model_stats["confMaxSigns"] = max(conf[res])
            confSigns = '({:.2f}, {:.2f})'.format(model_stats["confMinSigns"], model_stats["confMaxSigns"])
        else:
            model_stats["confMinSigns"] = None
            model_stats["confMaxSigns"] = None
            confSigns = ""


    # Pascal dataset
    else:
        # Get labels
        pascalLabels = pascal.labels()

        labelList = ["car", "bus", "motorcycle", "bicycle", "person"]
        statList = ["numCars", "numTrucksBuses", "numMotorCycles", "numBikes", "numPed"]

        for i in range(5):
            model_stats[statList[i]] = np.count_nonzero(labels == pascalLabels[labelList[i]])

        model_stats["numSign"] = 0 # Not available in PASCAL!

        # Extract confidences
        model_stats, confCars = getMinMaxConf(conf, labels, pascalLabels["car"], model_stats, "confMinCars", "confMaxCars")
        model_stats, confTrucksBuses = getMinMaxConf(conf, labels, pascalLabels["bus"], model_stats, "confMinTrucksBuses", "confMaxTrucksBuses")
        model_stats, confMotorCycles = getMinMaxConf(conf, labels, pascalLabels["motorcycle"], model_stats, "confMinMotorCycles", "confMaxMotorCycles")    
        model_stats, confBikes = getMinMaxConf(conf, labels, pascalLabels["bicycle"], model_stats, "confMinBikes", "confMaxBikes")
        model_stats, confPed = getMinMaxConf(conf, labels, pascalLabels["person"], model_stats, "confMinPed", "confMaxPed")
  
        # No signs in Pascal
        model_stats["confMinSigns"] = None
        model_stats["confMaxSigns"] = None
        confSigns = ""

    # Format text
    texts = ["cars: " + str(model_stats["numCars"]) + " " + confCars,
    "trucks, buses: " + str(model_stats["numTrucksBuses"]) + " " + confTrucksBuses,
    "motorcycles: " + str(model_stats["numMotorCycles"]) + " " + confMotorCycles,
    "bicycles: " + str(model_stats["numBikes"]) + " " + confBikes,
    "pedestrians: " + str(model_stats["numPed"]) + " " + confPed,
    "signs: " + str(model_stats["numSign"]) + " " + confSigns]

    # Place text
    for i in range(6):
        cv.putText(image, texts[i], (5, int(image.shape[0]/2)+20+20*i), constants.statsFormat()["font"], constants.statsFormat()["fontsize"], constants.statsFormat()["colour"], 1)
        
    return image, model_stats