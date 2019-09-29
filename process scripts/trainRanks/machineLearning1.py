import matplotlib.pyplot as plt
import cv2
from sklearn import datasets
from sklearn import svm
from sklearn import neighbors
from scipy import misc
import numpy as np

digits = datasets.load_digits()
knn = neighbors.KNeighborsClassifier()

clf = svm.SVC(gamma=0.1, C=100)

features, labels = digits.data, digits.target
clf.fit(features, labels)

img = cv2.imread('QUERY2.jpg', cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (8, 8), 0, 0)
img = np.asarray(img, dtype=np.float64)
rows, cols = img.shape
img = img.astype(digits.images.dtype)

query = []
for r in range(rows):
    for c in range(cols):
        query.append(abs(img[r][c])/16.0)
print query
query = np.asarray(query, dtype=np.float64)

print "Prediction:", clf.predict([query])

plt.gray()
plt.imshow(img, cmap=plt.cm.gray_r, interpolation="nearest")
plt.show()
#plt.imshow(digits.images[-4], cmap=plt.cm.gray_r, interpolation="nearest")
#plt.show()


"""
x_test = []
for row in img:
    for px in row:
        x_test.append(px)

print 'Prediction:', clf.predict([x_test])""""""

cv2.imshow('train', digits.target[-2])
cv2.imshow('test', img)

cv2.waitKey()
cv2.destroyWindow(0)
cap.release()
"""