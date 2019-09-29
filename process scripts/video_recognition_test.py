"""
live streaming, detecting contour with 4 corners, approximating corners
Updated on 15/11/18
"""
import cv2
import numpy as np

cap = cv2.VideoCapture(0)  # 0 = default camera on laptop

while True:
    ret, img = cap.read()  # ret is either True/False, frame displays camera frame

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # converting original image to greyscale
    blurred_img = cv2.GaussianBlur(gray, (3, 3), 0)  # applying blurring to greyscale image by 3 x 3 kernel
    _, binary_img = cv2.threshold(blurred_img, 150, 255, cv2.THRESH_BINARY_INV)  # applying binary threshold, either black/white

    # finds all contours, cv2.CHAIN_APPROX_NONE displays all contours
    _, contours, _ = cv2.findContours(binary_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    for contour in contours:
        area = cv2.contourArea(contour)
        #print area
        if area > 17000:  # set threshold, removes smaller contours
            cv2.drawContours(img, contour, -1, (0, 255, 0), 3)  # draws contour of all contours with over thresh area
            perimeter = cv2.arcLength(contour, True)  # calculates the arc length of contour
            # fix approx to extimate corners of more than one card, maybe put contour in a list
            approx = cv2.approxPolyDP(contour, 0.01 * perimeter, True)  # provides estimation of the corner, based on arc

    warpTest = []
    for i in approx:
        for j in i:
            print j
            x, y = j
            if len(approx) == 4:
                cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
                warpTest.append((x, y))

    cv2.imshow('Original Image', img)
    #cv2.imshow("blur", blurred_img)
    #cv2.imshow('Binary', binary_img)

    key = cv2.waitKey(20)  # sets a laptop key to waitKey
    if key == 27: # exit on ESC
        break

cv2.destroyWindow()
cap.release()
"""
    pts1 = np.float32([warpTest[0], warpTest[3], warpTest[1], warpTest[2]])
    pts2 = np.float32([[0, 0], [400, 0], [0, 600], [400, 600]])

    matrix = cv2.getPerspectiveTransform(pts1, pts2)  # similar to map function

    result = cv2.warpPerspective(img, matrix, (400, 600))  # warps image to dimension x & y
    cv2.imshow("perspective Transformation", result)
"""