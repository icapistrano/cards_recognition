import matplotlib.pyplot as plt
import cv2
import numpy as np

from sklearn import datasets
from sklearn.neural_network import MLPClassifier
from scipy import misc

digits = datasets.load_digits()
features = digits.data
labels = digits.target

clf = MLPClassifier()  #2, 4, 7,
clf.fit(features, labels)

img = misc.imread("Machine Learning/5.jpg")
img2 = cv2.imread("Nine.jpg")

img = misc.imresize(img, (8,8))
img2 = cv2.resize(img2, (8, 8))

img = img.astype(digits.images.dtype)
img2 = img2.astype(digits.images.dtype)

img = misc.bytescale(img, high=16, low=0)
img2 = misc.bytescale(img2, high=16, low=0)

y_test =[]
for eachRow in img2:
	for eachPixel in eachRow:
		y_test.append(sum(eachPixel)/3.0)
print(clf.predict([y_test]))


#plt.imshow(img, cmap=plt.cm.gray_r, interpolation="nearest")
#plt.show()
plt.imshow(img2, cmap=plt.cm.gray_r, interpolation="nearest")
plt.show()