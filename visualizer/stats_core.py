import cv2 as cv

def showStats(image, stats):
    """
    
    Args:
    image - 
    stats - List with core stats. 0 is fps, 1 is inference time, 2 inference time start, 3 inference time end

    Returns:
    Image - 
    """


    fps = stats[0]
    text = "fps: " + str(fps)
    cv.putText(image, text, (5, int(image.shape[0]/2)-20), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

    inference_time = stats[2] - stats[1]
    text = "Inference time: {:5.4f}s".format(inference_time)
    cv.putText(image, text, (5, int(image.shape[0]/2)), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

    return image