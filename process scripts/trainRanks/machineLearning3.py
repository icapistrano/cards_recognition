import matplotlib.pyplot as plt
import cv2
import numpy as np
from numpy import array

from sklearn import datasets
from sklearn.neural_network import MLPClassifier
from sklearn import datasets

img = cv2.imread("Two.jpg", cv2.IMREAD_UNCHANGED)
_, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV)

#kernel = np.ones((3,3),np.uint8)
#img = cv2.erode(img,kernel,iterations = 1)

#img = cv2.resize(img, (8, 8))
#img = img.astype(digits.data.dtype)
rows, cols = img.shape

ace_list = []
for r in range(rows):
    for c in range(cols):
        ace_list.append(img[r][c])
print ace_list

digits = datasets.load_digits()
x, y = digits.data, digits.target

target = ['Ace']
image = ace_list

clf = MLPClassifier()
clf.fit(target, image)

print (target)
print (digits.target)


#plt.imshow(img, cmap=plt.cm.gray_r, interpolation="nearest")
#plt.show()