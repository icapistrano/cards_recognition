"""
Script for recognising cards by suite & rank.
Camera must be positioned downwards, against a dark background
"""

# Import required libraries
from __future__ import division
import cv2
import numpy as np
import os


# Functions
def find_longContour(contours_list):  # function to find the longest contour in a given list
    max_so_far = cv2.arcLength(contours_list[0], True)  # sets perimeter max to the first contour
    max_pixels = contours_list[0]  # the coordinates of the first contour
    for contour in contours_list:
        perimeter = cv2.arcLength(contour, True)  # openCV function to find perimeter of a contour
        if perimeter > max_so_far:
            max_so_far = perimeter
            max_pixels = contour
    return [max_so_far, max_pixels]  # returning list [longest perimeter, and its coordinates]


def find_boundingRect(contour_coordinates, query_patch, resize):  # function to get minimum bounding area
    x, y, w, h = cv2.boundingRect(contour_coordinates)  # finds the minimum area, based on contour
    q_roi = query_patch[y:y + h, x:x + w]  # x, y = center, w= width, h= height
    q_resize = cv2.resize(q_roi, resize, 0, 0)  # same size as train images  #CHANGE THIS TO BE VARIANT
    return q_resize  # returns image in resized size


def find_bestMatch(filename, query):  # function to find the best match for suite & rank
    image_list = []  # creates a list with obtained images
    for file in filename:
        image_list.append(file)

    shape_dict = {}  # dictionary for acquiring suites, and its similarities with the query image
    for i, shape in enumerate(image_list):
        img = cv2.imread(image_list[i], cv2.IMREAD_UNCHANGED)
        res = cv2.matchTemplate(query, img, cv2.TM_CCOEFF_NORMED)  # template matching
        shape_dict[shape] = res[0][0]  # shape : values only, res = list within list
        best_matchVal = max(shape_dict.values())  # finds the maximum percentage
        best_shape = shape_dict.keys()[shape_dict.values().index(best_matchVal)]
    return [best_shape[11:-4], best_matchVal]  # return a list of best suite/rank and its highest score


def load_images(pathfile):
    return [os.path.join(pathfile, f) for f in os.listdir(pathfile) if f.endswith('jpg')]


cap = cv2.VideoCapture(1)  # 0 = default camera on laptop

while True:
    ret, frame = cap.read()  # ret is either True/False, frame displays camera frame

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # converting original image to greyscale
    blurred_frame = cv2.GaussianBlur(gray, (3, 3), 0)  # applying blurring to greyscale image by 3 x 3 kernel

    _, binary_frame = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # separates foreground & background

    # cv2.RETR_EXTERNAL removes child contours
    _, contours, _ = cv2.findContours(binary_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # finds all contours

    longest_contour = find_longContour(contours)

    approx = cv2.approxPolyDP(longest_contour[1], 0.01 * longest_contour[0], True)  # finds curves along the contour

    # sets colours to corners for trouble shooting & finding out where to transform
    RED = approx[0][0]
    YELLOW = approx[1][0]
    GREEN = approx[2][0]
    BLUE = approx[3][0]

    if len(approx) == 4:  # will not run if there are more/less than 4 corners in the frame
        # draws different coloured circles for each corner
        cv2.circle(frame, (RED[0], RED[1]), 5, (0, 0, 255), -1)  # red
        cv2.circle(frame, (BLUE[0], BLUE[1]), 5, (255, 0, 0), -1)  # blue
        cv2.circle(frame, (GREEN[0], GREEN[1]), 5, (0, 255, 0), -1)  # green
        cv2.circle(frame, (YELLOW[0], YELLOW[1]), 5, (0, 255, 255), -1)  # yellow

        # deals with orientation problem when warping image to a flat image, uses Pythagoras' theorem to switch corners
        distanceBG = np.sqrt((BLUE[0] - GREEN[0]) ** 2 + (BLUE[1] - GREEN[1]) ** 2)
        distanceBR = np.sqrt((BLUE[0] - RED[0]) ** 2 + (BLUE[1] - RED[1]) ** 2)

        if distanceBG < distanceBR:
            pts1 = np.float32([YELLOW, RED, GREEN, BLUE])  # top-left, top-right, bottom-left, bottom-right
        else:
            pts1 = np.float32([RED, BLUE, YELLOW, GREEN])  # top-left, top-right, bottom-left, bottom-right

        pts2 = np.float32([[0, 0], [400, 0], [0, 600], [400, 600]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)  # similar to map function
        flat_img = cv2.warpPerspective(gray, matrix, (400, 600))  # warps image by 400x600 flattened image

        patch = cv2.resize(flat_img[20:160, 0:60], (0, 0), fx=3, fy=3)
        #patch_grey = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)  # converting original image to greyscale
        _, binary_img = cv2.threshold(patch, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        query_rank = binary_img[0:240, 10:180]  # patch to extract rank
        query_suite = binary_img[240:420, 10:180]  # patch to extract suite

        # finds contours for rank & suite
        _, rank_contours, _ = cv2.findContours(query_rank, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        _, suite_contours, _ = cv2.findContours(query_suite, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # finds the longest contour in patches rank & suite
        rank_contour = find_longContour(rank_contours)
        suite_contour = find_longContour(suite_contours)

        # finds minimum bounding rectangle for rank & suite, resize scale
        bounded_qRank = find_boundingRect(rank_contour[1], query_rank, (70, 125))
        bounded_qSuite = find_boundingRect(suite_contour[1], query_suite, (70, 100))

        filename_rank = load_images(pathfile='trainRanks/')  # loads images within trainSuites
        filename_suites = load_images(pathfile='trainSuite/')  # loads images within trainSuites

        q_rank = find_bestMatch(filename_rank, bounded_qRank)
        q_suite = find_bestMatch(filename_suites, bounded_qSuite)

        if q_rank[1] and q_suite[1] < 0.65:
            cv2.putText(frame, 'Unknown Card', (350, 470), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        else:
            cv2.putText(frame, q_rank[0]+' of '+q_suite[0], (350, 470), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


        #cv2.imshow("Perspective Transformation", flat_img)
        cv2.imshow("rank", bounded_qRank)
        cv2.imshow("suite", bounded_qSuite)

    else:
        cv2.putText(frame, "No card detected", (350, 470), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        #print "No card detected"

    cv2.imshow('Original Frame', frame)
    #cv2.imshow('Binary Frame', binary_frame)

    key = cv2.waitKey(20)  # sets a laptop key to waitKey
    if key == 27:  # exit on ESC
        break

cv2.destroyWindow()
cap.release()