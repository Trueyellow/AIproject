# Rutger Introduction To Artificial Intelligence (CS 520) Final project
# test.py
# Final test program
# Date 05/02/2017
# Created by Kaixiang Huang, Yehan Wang

from Neural_Network import NeuralNetworkClassifier
from Naive_Bayesian_face import NaiveBayesClassifier
from Naive_Bayesian_digit import NaiveBayesClassifierDigit
from Perceptron import PerceptronClassifier
import numpy as np
import Feature
from samples import evaluate, loadDataFile, loadLabelsFile

fraction = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
number_of_train_data_digit = 5000
number_of_validation_data_digit = 1000
number_of_test_data_digit = 1000

number_of_train_data_face = 451
number_of_validation_data_face = 301
number_of_test_data_face = 150

digit_train_datas = loadDataFile("digitdata/trainingimages", number_of_train_data_digit, 28, 28)
digit_train_labels = loadLabelsFile("digitdata/traininglabels", number_of_train_data_digit)
digit_validation_datas = loadDataFile("digitdata/validationimages", number_of_validation_data_digit, 28, 28)
digit_validation_labels = loadLabelsFile("digitdata/validationlabels", number_of_validation_data_digit)
digit_test_datas = loadDataFile("digitdata/testimages", number_of_test_data_digit, 28, 28)
digit_test_labels = loadLabelsFile("digitdata/testlabels", number_of_test_data_digit)

face_train_datas = loadDataFile("facedata/facedatatrain", number_of_train_data_face, 60, 70)
face_train_labels = loadLabelsFile("facedata/facedatatrainlabels", number_of_train_data_face)
face_validation_datas = loadDataFile("facedata/facedatavalidation", number_of_validation_data_face, 60, 70)
face_validation_labels = loadLabelsFile("facedata/facedatavalidationlabels", number_of_validation_data_face)
face_test_datas = loadDataFile("facedata/facedatatest", number_of_test_data_face, 60, 70)
face_test_labels = loadLabelsFile("facedata/facedatatestlabels", number_of_test_data_face)


def NaiveBayes(data_type, train_data, train_label, test_data, test_label):
    if data_type == "digit":
        nb = NaiveBayesClassifierDigit(1)
    else:
        nb = NaiveBayesClassifier(1)
    nb.train(train_data, train_label)
    prediction = nb.classify(test_data)
    evaluate(prediction, test_label)


def Percepton(train_data, train_label, test_data, test_label):
    per = PerceptronClassifier()
    per.train(train_data, train_label, 30)
    prediction = per.classify(test_data)
    evaluate(prediction, test_label)


def NeuralNetwork(data_type, train_data, train_label, validation_data, validation_label, test_data, test_label):
    nn = NeuralNetworkClassifier(data_type)
    train_data = Feature.flatten_feature(train_data, 0)
    validation_data = Feature.flatten_feature(validation_data, 0)
    test_data = Feature.flatten_feature(test_data, 0)
    nn.train(train_data, train_label, validation_data, validation_label)
    prediction = nn.prediction(test_data)
    evaluate(prediction, test_label)


def Random_choice(fraction, data, label):
    train_mask = np.random.choice(len(data), int(fraction * len(data)))
    data = (np.array(data)[train_mask]).tolist()
    label = (np.array(label)[train_mask]).tolist()
    return data, label

if __name__ == "__main__":

    # --------------Generate Randomly data points with fraction
    # in case the training step will not be influenced by data's sequence-----------------------------------

    for fra in fraction:
        digit_train_data, digit_train_label = Random_choice(fra, digit_train_datas, digit_train_labels)
        digit_validation_data, digit_validation_label = Random_choice(fra, digit_validation_datas,
                                                                        digit_validation_labels)

        face_train_data, face_train_label = Random_choice(fra, face_train_datas, face_train_labels)

        print("\n----------------This is NaiveBayes classifier of digit with {} of train data, which we extract "
              "extra feature -- the number of closure cycles ---------------------------".format(fra))
        NaiveBayes("digit", digit_train_data, digit_train_label, digit_test_datas, digit_test_labels)
        print("\n----------------This is NaiveBayes classifier of face with {} of train data"
              "-------------------------".format(fra))
        NaiveBayes("face", face_train_data, face_train_label, face_test_datas, face_test_labels)

        print("\n----------------This is Percepton classifier of digit with {} of train data"
              "-------------------------".format(fra))
        Percepton(digit_train_data, digit_train_label, digit_test_datas, digit_test_labels)
        print("\n----------------This is Percepton classifier of face with {} of train data"
              "--------------------------".format(fra))
        Percepton(face_train_data, face_train_label, face_test_datas, face_test_labels)

        print("\n----------------This is NeuralNetwork classifier of digit with {} of train data"
              "---------------------------".format(fra))
        NeuralNetwork("digit", digit_train_data, digit_train_label, digit_validation_data,
                                digit_validation_label, digit_test_datas, digit_test_labels)

        print("\n----------------This is NeuralNetwork classifier of face with {} of train data"
              "---------------------------".format(fra))
        NeuralNetwork("face", face_train_data, face_train_label, face_validation_datas,
                                face_validation_labels, face_test_datas, face_test_labels)