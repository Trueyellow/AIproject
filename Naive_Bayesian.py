# Rutger Introduction To Artificial Intelligence (CS 520) Final project
# Naive Bayesian.py
# Date: 04/08/2017
# Created by Kaixiang Huang, Yehan Wang
# Based on the  http://inst.eecs.berkeley.edu/~cs188/sp11/projects/classification/naiveBayes.py

from samples import evaluate
import numpy as np
import math
import collections
import Feature
from samples import timecounter, loadDataFile, loadLabelsFile

class NaiveBayesClassifier():

    def __init__(self, k):
        self.k = k  # this is the smoothing parameter
        self.y_prior = {}
        self.y_Distribution = {}
        self.conditional_probabilities = {}

    def setSmoothing(self, k):
        """
        This is used by the main method to change the smoothing parameter before training.
        Do not modify this method.
        """
        self.k = k

    # function to calculate Prior Distribution -- P(Y) = C(Y)/n
    def calculate_prior_distribution(self, labels):

        self.y_Distribution = collections.Counter(labels)

        self.y_prior = {}

        for i in self.y_Distribution:
            self.y_prior[i] = float(self.y_Distribution[i] / len(labels))
        return self.y_prior

    # Function to calculate Conditional Probabilities-- P( F=fi\Y = y)
    def calculate_conditional_probabilities(self, image_data, labels):
        y = list(set(labels))
        c_fi_y = {}
        for i in range(len(labels)):
            label = labels[i]
            image_pixel = Feature.basicFeaturesExtract(image_data[i])
            if label not in c_fi_y:
                c_fi_y[label] = np.array(image_pixel)
            else:
                c_fi_y[label] += np.array(image_pixel)
        self.conditional_probabilities = {}
        c_FI_y = self.y_Distribution
        for label in c_fi_y:
            self.conditional_probabilities[label] = np.divide(c_fi_y[label] + self.k, float(c_FI_y[label] + 2 * self.k))

    def calculate_log_joint_probabilities(self, datum):
        """
        Returns the log-joint distribution over legal labels and the datum.
        """
        log_joint_probabilities = {}
        image_pixels = Feature.basicFeaturesExtract(datum)
        for label in self.y_Distribution:
            log_prior_distribution = math.log(self.y_prior[label])
            log_conditional_probabilities_1 = np.log(self.conditional_probabilities[label])
            log_conditional_probabilities_0 = np.log(1 - self.conditional_probabilities[label])

            log_joint_probabilities[label] = np.sum(np.array(image_pixels) * log_conditional_probabilities_1,
                                                    dtype=float)
            log_joint_probabilities[label] += np.sum(np.array(image_pixels) * log_conditional_probabilities_0,
                                                    dtype=float)
            log_joint_probabilities[label] += log_prior_distribution

        return log_joint_probabilities

    @timecounter
    def train(self, data, labels):
        self.calculate_prior_distribution(labels)
        self.calculate_conditional_probabilities(data, labels)

    def classify(self, testData):
        """
        Classify the data based on the posterior distribution over labels.
        """
        guesses = []
        self.posteriors = []  # Log posteriors are stored for later data analysis (autograder).
        for datum in testData:
            posterior = self.calculate_log_joint_probabilities(datum)
            guesses.append(max(posterior, key=(lambda key: posterior[key])))
            self.posteriors.append(posterior)
        return guesses

# n = 5000
# # items = loadDataFile("digitdata/trainingimages", n, 28, 28)
# # labels = loadLabelsFile("digitdata/traininglabels", n)
# # test_items = loadDataFile("digitdata/testimages", 500, 28, 28)
# # test_labels = loadLabelsFile("digitdata/testlabels", 500)
# items = loadDataFile("facedata/facedatatrain", n, 60, 70)
# labels = loadLabelsFile("facedata/facedatatrainlabels", n)
#
# test_items = loadDataFile("facedata/facedatatest", 301, 60, 70)
# test_labels = loadLabelsFile("facedata/facedatatestlabels", 301)
# print(len(items))
# nb = NaiveBayesClassifier(1)
# nb.train(items, labels)
#
# guess = nb.classify(test_items)
# evaluate(guess, test_labels)
# print(guess)
# print(test_labels)