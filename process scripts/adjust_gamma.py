"""
This is to change the threshold depending on total pixel value of the frame
"""

from __future__ import division
import cv2

def valmap(value, istart, istop, ostart, ostop):
    total = (ostart + (ostop - ostart) * ((value - istart) /(istop - istart)))
    return int(total)

cap = cv2.VideoCapture(0)  # 0 = default camera on laptop

while True:
    ret, img = cap.read()  # ret is either True/False, frame displays camera frame
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # converting original image to greyscale
    rows, cols = gray_img.shape

    px_list = []
    for r in range(rows):
        for c in range(cols):
            px = gray_img[r][c]
            px_list.append(px)
    total = sum(px_list)

    print valmap(total, 300000, 50000000, 0, 255)
    cv2.imshow('Original Image', gray_img)

    key = cv2.waitKey(1)  # sets a laptop key to waitKey
    if key == 27:  # exit on ESC
        break

cv2.destroyWindow()
cap.release()
"""
    px_list = []
    for r in range(rows):
        for c in range(cols):
            px = gray_img[r][c]
            px_list.append(px)
    total = sum(px_list)
    #print total

    #white = 45000000
    #black = 300000
"""