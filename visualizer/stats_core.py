import cv2 as cv

def showStats(image, stats):
    """
    Displays localization model independent stats in the image.
    The text is placed in the middle of the height and above.

    Args:
    image - Opencv image object in RGB
    stats - List with core stats. [fps, inference_time.start, inference_time.end]

    Returns:
    Image - Opencv image object in RGB with stats
    """

    fps = stats[0]
    text = "fps: {:2.0f}".format(fps)
    cv.putText(image, text, (5, int(image.shape[0]/2)-20), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

    inference_time = stats[2] - stats[1]
    text = "Inference time: {:5.4f}s".format(inference_time)
    cv.putText(image, text, (5, int(image.shape[0]/2)), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

    return image