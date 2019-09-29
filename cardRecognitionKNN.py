# Script for recognising rank & suits using KNN

# Import required libraries
import cv2
import numpy as np
import os
import pickle
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

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

#  function to predict class by using HOG as a feature descriptor and KNN as a classifier
def predict_query(filename):
    labels = {}  # dictionary to return value by index
    for index, suites in enumerate(['clubs', 'diamonds', 'hearts', 'spades', 'ace', 'two', 'three', 'four',
                                    'five', 'six', 'seven', 'eight', 'nine', 'ten', 'jack', 'queen', 'king']):
        labels.update({index: suites})  # append to dictionary with index: shape

    # process query the exact same way as training images
    query = cv2.resize(filename, (100, 100), fx=0, fy=0)

    # Histogram of Orientated Gradients - feature descriptor
    hog_image = hog(query, orientations=8, pixels_per_cell=(5, 5), cells_per_block=(5, 5), block_norm='L2-Hys')

    prediction = KNN_clf.predict([hog_image])  # predicts hog image, features in list
    label = prediction[0]  # comes in a list, so set label to the element with the first index, removes []
    return labels[label]

def load_images(pathfile):
    return [os.path.join(pathfile, f) for f in os.listdir(pathfile) if f.endswith('jpg')]  # pull all files with jpg

# Main
cap = cv2.VideoCapture(1)  # 0 = default camera on laptop
KNN_clf = pickle.load(open("KNN_model.pkl", "r"))  # getting trained model from current folder

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
    RED = approx[0][0]
    YELLOW = approx[1][0]
    GREEN = approx[2][0]
    BLUE = approx[-1][0]  # fixed bug of going out of range

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

        top_half = patch[20:240, 30:160]  # patch to isolate rank
        bottom_half = patch[240:400, 30:160]  # patch to isolate suite

        query_rank = predict_query(top_half)  # predict query
        query_suite = predict_query(bottom_half)  # predict suite
        cv2.putText(frame, query_rank + ' of ' + query_suite, (10, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 2)
    else:
        cv2.putText(frame, "No card detected... Place card within frame", (10, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 2)

    cv2.imshow('Card Detector', frame)

    key = cv2.waitKey(20)  # sets a laptop key to waitKey
    if key == 27:  # exit on ESC
        break

cv2.destroyWindow()
cap.release()