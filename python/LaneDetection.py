import numpy as np
from PIL import ImageGrab
import cv2
import time


def roi_mask(img, vertices):
  mask = np.zeros_like(img)
  mask_color = 255
  cv2.fillPoly(mask, vertices, mask_color)
  masked_img = cv2.bitwise_and(img, mask)
  return masked_img


# Hough transform parameters
rho = 3
theta = np.pi / 180
threshold = 10
min_line_length = 250
max_line_gap = 14


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def hough_lines(img, rho, theta, threshold, 
                min_line_len, max_line_gap):
  lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]),
                          minLineLength=min_line_len, 
                          maxLineGap=max_line_gap)
  line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
  draw_lines(line_img, lines)
  return line_img



def screen_record(): 
    last_time = time.time()
    while(True):
        printscreen = np.array(ImageGrab.grab(bbox=(10,10,1200,800))) 
        gray = cv2.cvtColor(printscreen, cv2.COLOR_RGB2GRAY)                 # Convert RGB into gray scale


        low_threshold = 150                                             # Canny edge detection low threshold
        high_threshold = 400                                    # Canny edge detection high threshold
        edges = cv2.Canny(gray, low_threshold, high_threshold)

        blur_ksize = 5  # Gaussian blur kernel size
        blur_edges = cv2.GaussianBlur(edges, (blur_ksize, blur_ksize), 0, 0)

        roi_vtx = np.array([[(0, printscreen.shape[0]), (400, 380), 
                     (800, 380), (printscreen.shape[1], printscreen.shape[0])]])
        roi_img = roi_mask(blur_edges, roi_vtx)      # Grab screen


        line_img = hough_lines(roi_img, rho, theta, threshold, min_line_length, max_line_gap)

        last_time = time.time()
        cv2.namedWindow("window",0)
        cv2.imshow('window', line_img)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

screen_record()