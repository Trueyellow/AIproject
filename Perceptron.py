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
                # ---- Updata weight for each class ---
                if prediction_label != true_label:
                    self.weights[true_label] = self.weights[true_label] + feature
                    self.weights[prediction_label] = self.weights[prediction_label] - feature

    def prediction(self, feature):
        scores = {}
        # use weights of each classes learned from training function to calculate scores of each classes
        for label in self.weights.keys():
            scores[label] = np.sum(self.weights[label]*feature)
        # find the predicted label with the highest score
        prediction = max(scores, key=(lambda key: scores[key]))
        return prediction

    def classify(self, testing_data):
        prediction_result = []
        for instance in range(len(testing_data)):
            feature = Feature.basicFeaturesExtract(testing_data[instance])
            prediction_result.append(self.prediction(feature))

        return prediction_result
