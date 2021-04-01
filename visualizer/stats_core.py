import cv2 as cv
from utils2 import constants

def show_stats(image, stats):
    """
    Displays localization model independent stats in the image.
    The text is placed in the middle of the height and above.

    Args:
    image - Opencv image object in RGB
    stats - List with core stats. [fps, inference_time.start, inference_time.end]

    Returns:
    Image - Opencv image object in RGB with stats
    """

    # Calculate stats
    fps = stats[0]
    inference_time = stats[2] - stats[1]

    # Format text
    texts = ["fps: {:2.0f}".format(fps), "Inference time: {:5.4f}s".format(inference_time)]

    # Put texts onto image
    for i in range(2):
        cv.putText(image, texts[i], (5, int(image.shape[0]/2)-20+20*i), constants.stats_format()["font"], 
        constants.stats_format()["fontsize"], constants.stats_format()["colour"], 1)
        
    return image