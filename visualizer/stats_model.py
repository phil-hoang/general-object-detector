import cv2 as cv
import numpy as np
from utils2 import constants
from visualizer import coco, pascal


def get_min_max_conf(conf, labels, label, model_stats, stats_min, stats_max):
    """
    Gets the min and max confidence for the given class.

    Args:
    conf        -- Torch tensor with confidences for each class.
    labels      -- Torch tensor with label indices of varying length.
    label       -- Integer with class label, as a dict key.
    model_stats -- Dictionary with stats. Keys: numCars, confMinCars, confMaxCars, ... 
    stats_min    -- String for min value name.
    stats_max    -- String for max value name.

    Returns:
    model_stats -- Dict with the updated stats.
    conf_class   -- String with the formatted min and max confidence.
    """
    if (len(conf[np.where(labels == label)])) > 0: 
        model_stats[stats_min] = min(conf[np.where(labels == label)])
        model_stats[stats_max] = max(conf[np.where(labels == label)])
        conf_class = '({:.2f}, {:.2f})'.format(model_stats[stats_min], model_stats[stats_max])
    else:
        model_stats[stats_min] = None
        model_stats[stats_max] = None
        conf_class = ""

    return model_stats, conf_class


def show_stats(image, model_type, labels, conf):
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
    model_stats = constants.model_stats()

    # COCO dataset
    coco_models = coco.supported_models()
        
    if (model_type in coco_models):
        # Get labels
        coco_labels = coco.labels()

        label_list = ["car", "motorcycle", "bicycle", "person"]
        stat_list = ["numCars", "numMotorCycles", "numBikes", "numPed"]

        for i in range(4):
            model_stats[stat_list[i]] = np.count_nonzero(labels == coco_labels[label_list[i]])

        model_stats["numTrucksBuses"] = np.count_nonzero(labels == coco_labels["truck"]) + np.count_nonzero(labels == coco_labels["bus"])
        model_stats["numSign"] = np.count_nonzero(labels == coco_labels["stopsign"]) + np.count_nonzero(labels == coco_labels["stoplight"])

        # Extract confidences abd 
        model_stats, conf_cars = get_min_max_conf(conf, labels, coco_labels["car"], model_stats, "confMinCars", "confMaxCars")
        model_stats, conf_motor_cycles = get_min_max_conf(conf, labels, coco_labels["motorcycle"], model_stats, "confMinMotorCycles", "confMaxMotorCycles")
        model_stats, conf_bikes = get_min_max_conf(conf, labels, coco_labels["bicycle"], model_stats, "confMinBikes", "confMaxBikes")
        model_stats, conf_ped = get_min_max_conf(conf, labels, coco_labels["person"], model_stats, "confMinPed", "confMaxPed")

        # Confidence for summarized classes. Currently they have to be handled seperatly.
        if (len(conf[np.where(labels == coco_labels["truck"])]) or len(conf[np.where(labels == coco_labels["bus"])])) > 0:
            res = np.append(np.where(labels == coco_labels["truck"]), np.where(labels == coco_labels["bus"]))
            model_stats["confMinTrucksBuses"] = min(conf[res])
            model_stats["confMaxTrucksBuses"] = max(conf[res])
            conf_trucks_buses = '({:.2f}, {:.2f})'.format(model_stats["confMinTrucksBuses"], model_stats["confMaxTrucksBuses"])
        else:
            model_stats["confMinTrucksBuses"] = None
            model_stats["confMaxTrucksBuses"] = None
            conf_trucks_buses = ""

        if (len(conf[np.where(labels == coco_labels["stopsign"])]) or len(conf[np.where(labels == coco_labels["stoplight"])])) > 0:
            res = np.append(np.where(labels == coco_labels["stopsign"]), np.where(labels == coco_labels["stoplight"]))
            model_stats["confMinSigns"] = min(conf[res])    
            model_stats["confMaxSigns"] = max(conf[res])
            conf_signs = '({:.2f}, {:.2f})'.format(model_stats["confMinSigns"], model_stats["confMaxSigns"])
        else:
            model_stats["confMinSigns"] = None
            model_stats["confMaxSigns"] = None
            conf_signs = ""


    # Pascal dataset
    else:
        # Get labels
        pascal_labels = pascal.labels()

        label_list = ["car", "bus", "motorcycle", "bicycle", "person"]
        stat_list = ["numCars", "numTrucksBuses", "numMotorCycles", "numBikes", "numPed"]

        for i in range(5):
            model_stats[stat_list[i]] = np.count_nonzero(labels == pascal_labels[label_list[i]])

        model_stats["numSign"] = 0 # Not available in PASCAL!

        # Extract confidences
        model_stats, conf_cars = get_min_max_conf(conf, labels, pascal_labels["car"], model_stats, "confMinCars", "confMaxCars")
        model_stats, conf_trucks_buses = get_min_max_conf(conf, labels, pascal_labels["bus"], model_stats, "confMinTrucksBuses", "confMaxTrucksBuses")
        model_stats, conf_motor_cycles = get_min_max_conf(conf, labels, pascal_labels["motorcycle"], model_stats, "confMinMotorCycles", "confMaxMotorCycles")    
        model_stats, conf_bikes = get_min_max_conf(conf, labels, pascal_labels["bicycle"], model_stats, "confMinBikes", "confMaxBikes")
        model_stats, conf_ped = get_min_max_conf(conf, labels, pascal_labels["person"], model_stats, "confMinPed", "confMaxPed")
  
        # No signs in Pascal
        model_stats["confMinSigns"] = None
        model_stats["confMaxSigns"] = None
        conf_signs = ""

    # Format text
    texts = ["cars: " + str(model_stats["numCars"]) + " " + conf_cars,
    "trucks, buses: " + str(model_stats["numTrucksBuses"]) + " " + conf_trucks_buses,
    "motorcycles: " + str(model_stats["numMotorCycles"]) + " " + conf_motor_cycles,
    "bicycles: " + str(model_stats["numBikes"]) + " " + conf_bikes,
    "pedestrians: " + str(model_stats["numPed"]) + " " + conf_ped,
    "signs: " + str(model_stats["numSign"]) + " " + conf_signs]

    # Place text
    for i in range(6):
        cv.putText(image, texts[i], (5, int(image.shape[0]/2)+20+20*i), constants.stats_format()["font"], constants.stats_format()["fontsize"], constants.stats_format()["colour"], 1)
        
    return image, model_stats