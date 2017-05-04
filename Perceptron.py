# Rutger Introduction To Artificial Intelligence (CS 520) Final project
# Perceptron.py
# Date: 04/09/2017
# Created by Kaixiang Huang, Yehan Wang
# Based on the  http://inst.eecs.berkeley.edu/~cs188/sp11/projects/classification/perceptron.py
# Perceptron implementation

import numpy as np
import Feature
from samples import timecounter


class PerceptronClassifier:

    """
    Perceptron classifier.
    """

    def __init__(self):
        self.weights = {}

    @timecounter
    def train(self, training_data, training_labels, iterations):
        weight = np.zeros((training_data[0].height, training_data[0].width))

        for i in training_labels:
            if i not in self.weights:
                self.weights[i] = weight

        for it in range(iterations):
            for instance_number in range(len(training_labels)):
                feature = Feature.basicFeaturesExtract(training_data[instance_number])
                true_label = training_labels[instance_number]
                prediction_label = self.prediction(feature)
                if prediction_label != true_label:
                    self.weights[true_label] = self.weights[true_label] + feature
                    self.weights[prediction_label] = self.weights[prediction_label] - feature



    def prediction(self, feature):
        scores = {}
        for label in self.weights.keys():
            scores[label] = np.sum(self.weights[label]*feature)
        prediction = max(scores, key=(lambda key: scores[key]))
        return prediction

    def classify(self, testing_data):
        predition_result = []
        for instance in range(len(testing_data)):
            feature = Feature.basicFeaturesExtract(testing_data[instance])
            predition_result.append(self.prediction(feature))

        return predition_result


# n = 5000
# # items = loadDataFile("digitdata/trainingimages", n, 28, 28)
# # labels = loadLabelsFile("digitdata/traininglabels", n)
# # test_items = loadDataFile("digitdata/testimages", 50, 28, 28)
# # test_labels = loadLabelsFile("digitdata/testlabels", 50)
# items = loadDataFile("facedata/facedatatrain", n, 60, 70)
# labels = loadLabelsFile("facedata/facedatatrainlabels", n)
#
# test_items = loadDataFile("facedata/facedatatest", 150, 60, 70)
# test_labels = loadLabelsFile("facedata/facedatatestlabels", 150)
# per = PerceptronClassifier()
# per.train(items, labels, 200)
# guess = per.classify(test_items)
# evaluate(guess, test_labels)
# print(guess)
# print(test_labels)