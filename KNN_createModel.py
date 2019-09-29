"""
Script for recognising classes using KNN.
"""

import os
import cv2
import numpy as np
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import pickle

def load_images(pathfile):
    return [os.path.join(pathfile, f) for f in os.listdir(pathfile)]

features = []  # store features of each images
labels = []  # label numbers by corresponding image features
guess = {}  # dictionary for linking labels and shapes later on..
for index, shape in enumerate(['clubs', 'diamonds', 'hearts', 'spades', 'ace', 'two', 'three', 'four',
                                'five', 'six', 'seven', 'eight', 'nine', 'ten', 'jack', 'queen', 'king']):

    filename = load_images(pathfile='dataset/' + shape)  # loads images within folder of shape
    guess.update({index: shape})  # append to dictionary with index: shape

    for images in filename:
        image = cv2.imread(images, cv2.IMREAD_GRAYSCALE)  # read each image
        img = cv2.resize(image, (100, 100), 0, 0)  # resize images to exact dimensions

        # Histogram of Orientated Gradients - feature descriptor
        hog_image = hog(img, orientations=8, pixels_per_cell=(5, 5), cells_per_block=(5, 5), block_norm='L2-Hys')

        features.append(hog_image)
        labels.append(index)

class_shape = guess.values()

features = np.array(features, dtype='float64')  # required to fit

# splitting images by 25% of total samples, random to remove 'bias'
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.25, random_state=10)

# KNN, odd number so a class always wins
# takes 3 closest neighbours, and classify itself by majority
clf = KNeighborsClassifier(n_neighbors= 3)

clf.fit(X_train, y_train)  # training


# pickle.dump(clf, open('KNN_model.pkl', 'wb'))  # save the model for future use

"""
# testing
# process query the exact same way as before..
query = cv2.imread('dataset/hearts/img_0.jpg', cv2.IMREAD_GRAYSCALE)
query = cv2.resize(query, (100, 100), fx=0, fy=0)
hog_image = hog(query, orientations=8, pixels_per_cell=(5, 5), cells_per_block=(5, 5), block_norm='L2-Hys')

prediction = clf.predict([hog_image])  # predicts hog image, only one image
label = prediction[0]  # comes in a list, so set label to the element with the first index
print class_shape[label]  # print the shape with label

"""