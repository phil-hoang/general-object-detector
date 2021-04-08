"""
Constants which are used in the program.

For constants related to the COCO and Pascal dataset, see their modules in visualizer.
"""

import cv2 as cv

def box_colours():
    box_colours = {"motor": (255, 0, 0), "person": (0, 0, 255), "bike": (255,165,0), "signs": (0, 255, 0), "other": (220, 220, 220)}

    return box_colours


def stats_format():
    """
    Returns a dict to select font type, font size and text colour.
    """
    vis = {"font": cv.FONT_HERSHEY_SIMPLEX, "fontsize": 0.7, "colour": (255,255,255)}

    return vis


def model_stats():
    """
    Stats for the model. Is used to both diplay stats as well as to log the information.
    Returns a dictionary with empty value lists and the set keys.
    """
    keys = ["numCars", "confMinCars", "confMaxCars",
            "numTrucksBuses", "confMinTrucksBuses", "confMaxTrucksBuses",
            "numMotorCycles", "confMinMotorCycles", "confMaxMotorCycles",
            "numBikes", "confMinBikes", "confMaxBikes",
            "numPed", "confMinPed", "confMaxPed",
            "numSign", "confMinSign", "confMaxSign"]

    # Initialize dict
    stats = {new_list: [] for new_list in keys}

    return stats
