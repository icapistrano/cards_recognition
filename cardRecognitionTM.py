# Script for recognising ranks & suit using Template Matching

# Import required libraries
import cv2
import numpy as np
import os

# Functions

#  function to find the longest contour in a list. Returns the longest perimeter & its cartesian coordinates
def find_longContour(contours_list):
    max_so_far = cv2.arcLength(contours_list[0], True)  # cv2 function to find the perimeter & set max to the 1st in list
    max_pixels = contours_list[0]  # coordinates assigned to variable
    for contour in contours_list:
        perimeter = cv2.arcLength(contour, True)  # go through each contour within list and find its perimeter
        if perimeter > max_so_far:  # if perimeter is larger, assign new perimeter and its coordinates
            max_so_far = perimeter
            max_pixels = contour
    return [max_so_far, max_pixels]

# function to find the minimum bounding rectangle and resize to required size
def find_boundingRect(contour_coordinates, query_patch, resize):
    x, y, w, h = cv2.boundingRect(contour_coordinates)  # function to find bounding box
    q_roi = query_patch[y:y + h, x:x + w]  # x, y= center coordinates, w= width, h= height
    q_resize = cv2.resize(q_roi, resize, 0, 0)  # same size as the training images
    return q_resize

# function to find the best match for suite & rank, return best match and its probability
def find_bestMatch(filename, query):
    image_list = []
    for file in filename:
        image_list.append(file)  # go through each images & append to list

    shape_dict = {}
    for i, shape in enumerate(image_list):
        img = cv2.imread(image_list[i], cv2.IMREAD_UNCHANGED)
        res = cv2.matchTemplate(query, img, cv2.TM_CCOEFF_NORMED)  # template match query acquired against all images
        shape_dict[shape] = res[0][0]  # assign shape to probability, res is a list of list
        best_matchVal = max(shape_dict.values())  # finds the maximum percentage in dictionary values
        best_shape = shape_dict.keys()[shape_dict.values().index(best_matchVal)]  # return dictionary key based on max dictionary value
    return [best_shape[11:-4], best_matchVal]


def load_images(pathfile):
    return [os.path.join(pathfile, f) for f in os.listdir(pathfile) if f.endswith('jpg')]  # pull all files with jpg







#  Main
cap = cv2.VideoCapture(1)  # 0 = default camera on laptop
counter = 0
suite_list = []
while True:
    ret, frame = cap.read()  # ret is either True/False, frame displays camera frame

    # processing the video
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # converting original image to grey
    blurred_frame = cv2.GaussianBlur(gray, (3, 3), 0)  # applying blurring to greyscale image by 3 x 3 kernel
    _, binary_frame = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # separates foreground & background

    # function to find all contours, cv2.RETR_EXTERNAL removes child contours
    _, contours, _ = cv2.findContours(binary_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    longest_contour = find_longContour(contours)  # find longest contour and assigned to new variable

    approx = cv2.approxPolyDP(longest_contour[1], 0.01 * longest_contour[0], True)  # estimate corner in contour

    # sets colours to corners to trouble shoot orientation problems & find out corners for flat perspective
    RED = (approx[0][0])
    YELLOW = (approx[1][0])
    GREEN = (approx[2][0])
    BLUE = (approx[-1][0])  # fixed bug of going out of range

    cv2.putText(frame, 'Press ESC to exit', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
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
            #  portrait
            pts1 = np.float32([YELLOW, RED, GREEN, BLUE])  # top-left, top-right, bottom-left, bottom-right
        else:
            #  landscape
            pts1 = np.float32([RED, BLUE, YELLOW, GREEN])  # top-left, top-right, bottom-left, bottom-right

        pts2 = np.float32([[0, 0], [400, 0], [0, 600], [400, 600]])  # set wanted size
        matrix = cv2.getPerspectiveTransform(pts1, pts2)  # similar to map function
        flat_img = cv2.warpPerspective(gray, matrix, (400, 600))  # warps image by 400x600 flattened image

        patch = cv2.resize(flat_img[20:160, 0:60], (0, 0), fx=3, fy=3)  # resize with scalar zoom

        # threshold corner of flat image & apply threshold. Need inverse binary to flip black & white to match training images
        _, binary_img = cv2.threshold(patch, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        query_rank = binary_img[0:240, 10:180]  # patch to isolate rank
        query_suite = binary_img[240:420, 10:180]  # patch to isolate suite

        # finds contours for rank & suite
        _, rank_contours, _ = cv2.findContours(query_rank, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        _, suite_contours, _ = cv2.findContours(query_suite, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # finds the longest contour in patches rank & suite
        rank_contour = find_longContour(rank_contours)
        suite_contour = find_longContour(suite_contours)

        # finds minimum bounding rectangle for rank & suite, resize scale
        bounded_qRank = find_boundingRect(rank_contour[1], query_rank, (70, 125))
        bounded_qSuite = find_boundingRect(suite_contour[1], query_suite, (70, 100))

        # load images within path file
        filename_rank = load_images(pathfile='trainRanks/')
        filename_suites = load_images(pathfile='trainSuite/')

        # find best match by template matching
        q_rank = find_bestMatch(filename_rank, bounded_qRank)
        q_suite = find_bestMatch(filename_suites, bounded_qSuite)

        card_list = []
        if q_rank[1] and q_suite[1] < 0.5:  # if returned probability is less than threshold value
            cv2.putText(frame, 'Unknown Card', (10, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 2)

        else:
            cv2.putText(frame, q_rank[0]+' of '+q_suite[0], (10, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 2)
            current_suite = q_suite[0]


    else:
        cv2.putText(frame, "No card detected... Place card within frame", (10, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 2)
    cv2.imshow('Card Detector', frame)
    suite_list.append(current_suite)
    print suite_list


    key = cv2.waitKey(20)  # sets a laptop key to waitKey
    if key == 27:  # exit on ESC
        break

cv2.destroyWindow()
cap.release()