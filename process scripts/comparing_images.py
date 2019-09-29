"""
Template matching
Updated on 24/11/18
"""

# Import libraries required
import cv2
import numpy as np
import os

def load_images(pathfile):
    return [os.path.join(pathfile, f) for f in os.listdir(pathfile) if f.endswith('jpg')]

filenames = load_images(pathfile = 'trainSuite/')  # loads images within trainSuites

query = cv2.imread('queryRank.jpg', cv2.IMREAD_UNCHANGED)  # read query obtained in main script

images = []  # creates a list with obtained images
for file in filenames:
    images.append(file)
print images

suite_dict = {}  # dictionary for acquiring suites, and its similarities with the query image
for i, suite in enumerate(images):
    img = cv2.imread(images[i], cv2.IMREAD_UNCHANGED)
    res = cv2.matchTemplate(query, img, cv2.TM_CCOEFF_NORMED)  # template matching
    suite_dict[suite] = res[0][0]  # suites : values only, res = list within list
    best_match = max(suite_dict.values())  # finds the maximum percentage
    best_suite = suite_dict.keys()[suite_dict.values().index(best_match)]
print res[0], best_suite[11:-4]



#cv2.imshow('query', query)
#cv2.waitKey()
#cv2.destroyAllWindows()