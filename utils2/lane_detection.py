import cv2 as cv
import numpy as np

def display_lines(image, lines):
    line_image = image
    try:
        line_image = np.zeros_like(image)
        if lines is not None:
            for x1, y1, x2, y2 in lines:
                cv.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    except: TypeError

    return line_image


def create_coordinates(image, line_parameters):
    x1 = y1 = x2 = y2 = 0
    try:
        slope, intercept = line_parameters
        y1 = image.shape[0]
        y2 = int(y1 * (7/10))
        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)
    except: TypeError
        

    return np.array([x1, y1, x2, y2])


def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    left_line = 0
    right_line = 0
    try:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)

            # It will fit the polynomial and the intercept and slope
            parameters = np.polyfit((x1, x2), (y1, y2), 1)
            slope = parameters[0]
            intercept = parameters[1]
            if slope < 0:
                left_fit.append((slope, intercept))
            else:
                right_fit.append((slope, intercept))
        left_fit_average = np.average(left_fit, axis=0)
        right_fit_average = np.average(right_fit, axis=0)
        left_line = create_coordinates(image, left_fit_average)
        right_line = create_coordinates(image, right_fit_average)
    except: TypeError

    return np.array([left_line, right_line])

def roi(image):
    height = image.shape[0]

    # Input array of arrays. Dim are (,,height of triangle tip)
    offset_bottom = 50
    # Offsets in percent
    offset_left = 0.05
    offset_right = 0.05
    offset_tip_y = 0.65
    offset_tip_from_centre = -0.07 # - to left, + to right
    
    # Make polygon
    polygons = np.array([[((int(image.shape[1] * offset_left)), height-offset_bottom), 
    (image.shape[1] - (int(image.shape[1] * offset_right)), height-offset_bottom), 
    (int((image.shape[1] - (int(image.shape[1] * offset_right)) + (int(image.shape[1] * offset_left)))/2) + (int((image.shape[1] // 2) * offset_tip_from_centre)), (int(image.shape[0] * offset_tip_y))) ]])
    
    # Background
    mask = np.zeros_like(image)

    # Fill the poly-function deals with multiple polygons
    cv.fillPoly(mask, polygons, 255)

    # Bitwise operation between canny image and mask image
    masked_image = cv.bitwise_and(image, mask)

    return masked_image

def canny_edge_detector(image):
    # Convert image to gray scale
    #gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    gray_image = image
    # Reduce noise from the image
    blur = cv.GaussianBlur(gray_image, (5, 5), 0)
    canny = cv.Canny(blur, 100, 150)

    return canny


def detect(image):
    """
    Detects lanes using a Hough transform

    Args:
    image   -- Opencv colour image

    Returns:
    iamge   -- Opencv image with lanes
    """

    img_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # Detect edges using Canny algorithm
    canny_image = canny_edge_detector(img_gray)
    
    # Crop image to region of interest
    cropped_image  = roi(canny_image)

    # Find lane lines
    lines = cv.HoughLinesP(cropped_image, 1, np.pi/180, 40, np.array([]), minLineLength = 15, maxLineGap = 10)
    averaged_lines = average_slope_intercept(image, lines)
    img_lines = display_lines(image, averaged_lines)
    image_with_lines = cv.addWeighted(image, 0.8, img_lines, 1, 1)

    return image_with_lines