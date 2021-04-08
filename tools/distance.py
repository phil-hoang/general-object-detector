import numpy as np


def getDistance(boxes):
    """
    
    Args:
    boxes       -- List of coordinates of the top left and bottom right of the bounding box ordered as [(x1, y1, x2, y2)]
    """
    focal = 50 # Camera constant [mm]
    width = 2000 # known width [mm]

    i = 0
    box = boxes[i, :]
    x1 = box[0]
    x2 = box[2]

    pixels = np.abs(x2 - x1)

    
    distance = (focal / pixels) * width

    return distance